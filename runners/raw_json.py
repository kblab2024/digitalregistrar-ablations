"""
Cell C — DSPy-less, raw LLM with JSON mode.

The entire pipeline runs without DSPy. We make at most two raw LLM
calls per report via the OpenAI-compatible chat API:

    Call 1 — classify: {cancer_excision_report, cancer_category}
    Call 2 — extract:  flat cancer_data dict matching the organ schema

Both calls use `response_format={"type": "json_object"}`. For the
extraction call the JSON schema is inlined in the system prompt (same
schema the parent pipeline emits at runtime, derived from
`schemas/schema_builder.py`).

Local models go through Ollama's OpenAI-compatible endpoint
(`http://localhost:11434/v1`), so the same `OpenAI` client works for
both `gpt-oss:20b` and `gpt-4-turbo`.

Usage:
    # Cell C × gpt-oss:20b
    python runners/raw_json.py --model gpt-oss:20b \\
        --api-base http://localhost:11434/v1 \\
        --out results/raw_json_gpt-oss

    # Cell C × gpt-4-turbo
    OPENAI_API_KEY=... python runners/raw_json.py --model gpt-4-turbo \\
        --out results/raw_json_gpt4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402
from schemas.schema_builder import (  # noqa: E402
    describe_field_list,
    flatten_schema_for_prompt,
    load_organ_schema,
    validate_cancer_data,
)

SPLITS_PATH = ROOT.parent / "digitalregistrar-benchmarks" / "data" / "splits.json"

CANCER_CATEGORIES = [
    "stomach", "colorectal", "breast", "esophagus", "lung", "prostate",
    "thyroid", "pancreas", "cervix", "liver", "others",
]

CLASSIFY_SYSTEM = """\
You are a cancer registrar. Given a pathology report, decide:
1. whether the report documents a PRIMARY cancer excision eligible for
   cancer-registry entry (false if no viable tumor remains after
   excision, or if the finding is carcinoma in situ / high-grade
   dysplasia only); and
2. which organ the primary cancer arises from.

Return ONLY a JSON object with this exact shape:

  {
    "cancer_excision_report": true | false,
    "cancer_category": """ + " | ".join(f'"{c}"' for c in CANCER_CATEGORIES) + """ | null,
    "cancer_category_others_description": string | null
  }

`cancer_category` must be null if `cancer_excision_report` is false.
`cancer_category_others_description` is non-null only when
`cancer_category` is "others".
"""

EXTRACT_SYSTEM_TEMPLATE = """\
You are a cancer registrar. Extract structured fields from the given
{organ} cancer excision report.

Return ONLY a JSON object matching the field definitions below. Include
every field in the output, using null when the field is not present in
the report. Do not invent values. Do not omit fields. Do not add extra
fields.

Schema (field_name (type): description):
{field_list}
"""


class RawJSONRunner:
    def __init__(self, model: str, api_key: str | None,
                 api_base: str | None):
        self.model = model
        kwargs = {"api_key": api_key or "EMPTY"}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.validation_retries = 0

    def _chat(self, system: str, user: str,
              schema_hint: dict | None = None) -> dict:
        """One JSON-mode chat call. `schema_hint` is passed for models
        that support `response_format={type: json_schema}` (OpenAI
        gpt-4-turbo and newer); falls back to `json_object` otherwise."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        except Exception as e:
            # Some backends (older Ollama versions) reject response_format.
            # Retry without the constraint.
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                        system + "\n\nIMPORTANT: respond with a single "
                                 "JSON object only — no prose, no code fences."},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
        text = resp.choices[0].message.content or "{}"
        return _parse_json_best_effort(text)

    def classify(self, report: str) -> dict:
        return self._chat(CLASSIFY_SYSTEM, report)

    def extract(self, report: str, organ: str) -> tuple[dict, list[str]]:
        schema = flatten_schema_for_prompt(load_organ_schema(organ))
        system = EXTRACT_SYSTEM_TEMPLATE.format(
            organ=organ, field_list=describe_field_list(schema))
        data = self._chat(system, report, schema_hint=schema)
        errors = validate_cancer_data(organ, data)
        # One retry with errors surfaced to the model.
        if errors and len(errors) < 20:
            self.validation_retries += 1
            repair_user = (
                f"The previous output had these schema errors:\n"
                + "\n".join(f"  - {e}" for e in errors[:20])
                + f"\n\nFix them and return the corrected JSON only.\n\n"
                f"Original report:\n{report}"
            )
            data = self._chat(system, repair_user, schema_hint=schema)
            errors = validate_cancer_data(organ, data)
        return data, errors

    def run_case(self, report: str) -> dict:
        cls = self.classify(report)
        if not cls.get("cancer_excision_report"):
            return {"cancer_excision_report": False,
                    "cancer_category": None, "cancer_data": {}}
        organ = cls.get("cancer_category")
        out = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                cls.get("cancer_category_others_description"),
            "cancer_data": {},
        }
        if organ in {"others", None} or not (ROOT.parent / "digitalregistrar-annotation"
                                             / "schemas" / f"{organ}.json").exists():
            return out
        cancer_data, errors = self.extract(report, organ)
        out["cancer_data"] = cancer_data
        if errors:
            out["_schema_errors"] = errors[:20]
        return out


def _parse_json_best_effort(text: str) -> dict:
    """Strip ``` fences, extract the largest {...} block, parse."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fence.
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: pull out the outermost braces.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_parse_error": text[:500]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model id (e.g. 'gpt-4-turbo' or 'gpt-oss:20b')")
    ap.add_argument("--api-base", default=None,
                    help="for local Ollama: http://localhost:11434/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Local Ollama default.
    api_base = args.api_base
    api_key = os.environ.get("OPENAI_API_KEY")
    if args.model.startswith(("gpt-oss", "gemma", "qwen", "phi", "llama")):
        api_base = api_base or "http://localhost:11434/v1"
        api_key = api_key or "ollama"

    runner = RawJSONRunner(args.model, api_key, api_base)

    with SPLITS_PATH.open(encoding="utf-8") as f:
        cases = json.load(f)["test"]
    if args.limit:
        cases = cases[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = []
    for case in cases:
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        try:
            result = runner.run_case(report)
        except Exception as e:
            result = {"_error": str(e)}
        elapsed = time.perf_counter() - t0
        with (out_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ledger.append({"id": case["id"], "elapsed_s": elapsed})
        print(f"  [{case['id']}] {elapsed:.1f}s")

    with (out_dir / "_ledger.json").open("w", encoding="utf-8") as f:
        json.dump({"model": args.model,
                   "validation_retries": runner.validation_retries,
                   "runs": ledger}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
