"""
Cell B — DSPy + monolithic (one big signature per organ).

Runs the same top-level pipeline as the parent project — `is_cancer`
routing, optional `ReportJsonize`, then **one** organ-specific DSPy
signature (instead of the 5–7 in the modular baseline).

Supports both backbone models referenced in the parent `model_list`
(`gpt` = local gpt-oss:20b via Ollama, `gpt4` = openai/gpt-4-turbo).

Usage:
    # Cell B × gpt-oss (local Ollama)
    python runners/dspy_monolithic.py --model gpt --out results/dspy_monolithic_gpt-oss

    # Cell B × gpt-4-turbo
    OPENAI_API_KEY=... python runners/dspy_monolithic.py \\
        --model gpt4 --out results/dspy_monolithic_gpt4

    # Skip the ReportJsonize intermediate step (supplementary variant)
    python runners/dspy_monolithic.py --model gpt --skip-jsonize \\
        --out results/dspy_monolithic_gpt-oss_nojsonize
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "digitalregistrar"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))

import dspy  # noqa: E402
from models.common import autoconf_dspy, is_cancer, ReportJsonize, model_list  # noqa: E402
from models.modellist import organmodels  # noqa: E402
from util.predictiondump import dump_prediction_plain  # noqa: E402
from signatures.monolithic import get_monolithic_signature  # noqa: E402

SPLITS_PATH = ROOT.parent / "digitalregistrar-benchmarks" / "data" / "splits.json"


class MonolithicPipeline(dspy.Module):
    """Drop-in replacement for `CancerPipeline` with per-organ signatures
    collapsed into a single monolithic signature."""

    def __init__(self, skip_jsonize: bool = False):
        super().__init__()
        self.skip_jsonize = skip_jsonize
        self.analyzer_is_cancer = dspy.Predict(is_cancer)
        self.jsonize = dspy.Predict(ReportJsonize)
        # Lazy-build monolithic predictors keyed by organ.
        self._organ_predictors: dict[str, dspy.Predict] = {}

    def _get_organ_predictor(self, organ: str) -> dspy.Predict:
        if organ not in self._organ_predictors:
            sig = get_monolithic_signature(organ)
            self._organ_predictors[organ] = dspy.Predict(sig)
        return self._organ_predictors[organ]

    def forward(self, report: str, logger: logging.Logger,
                fname: str = "") -> dict:
        logger.info(f"[monolithic] {fname}")
        paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]

        context_response = self.analyzer_is_cancer(report=paragraphs)
        if not context_response.cancer_excision_report:
            return {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_data": {},
            }

        organ = context_response.cancer_category
        out = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                context_response.cancer_category_others_description,
            "cancer_data": {},
        }

        if organ not in organmodels:
            # "others" or unknown — skip the organ-specific stage entirely.
            return out

        report_jsonized: dict = {}
        if not self.skip_jsonize:
            try:
                jr = self.jsonize(report=paragraphs, cancer_category=organ)
                report_jsonized = jr.output or {}
            except Exception as e:
                logger.warning(f"jsonize failed for {fname}: {e}")

        try:
            predictor = self._get_organ_predictor(organ)
            organ_response = predictor(
                report=paragraphs, report_jsonized=report_jsonized)
            out["cancer_data"] = dump_prediction_plain(organ_response)
        except Exception as e:
            logger.error(f"monolithic {organ} failed for {fname}: {e}")
            out["_error"] = str(e)

        return out


def _load_split() -> list[dict]:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} not found. Run "
            f"`python ../digitalregistrar-benchmarks/data/split.py` first.")
    with SPLITS_PATH.open(encoding="utf-8") as f:
        return json.load(f)["test"]


def _setup_model(model_key: str) -> None:
    """Configure DSPy LM. `model_key` is either a key from the parent
    project's `model_list` (local Ollama models) or "gpt4" for OpenAI."""
    if model_key == "gpt4":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        lm = dspy.LM(
            model="openai/gpt-4-turbo",
            api_key=api_key,
            max_tokens=16384,
            temperature=0.0,
        )
        dspy.configure(lm=lm)
        return
    if model_key not in model_list:
        raise ValueError(
            f"Unknown model key {model_key!r}. Available: "
            f"{list(model_list.keys()) + ['gpt4']}")
    autoconf_dspy(model_key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="key in parent's model_list (e.g. 'gpt') or 'gpt4'")
    ap.add_argument("--out", required=True,
                    help="output directory (per-case JSON)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-jsonize", action="store_true",
                    help="ablation-of-ablation: also drop ReportJsonize")
    args = ap.parse_args()

    _setup_model(args.model)
    pipe = MonolithicPipeline(skip_jsonize=args.skip_jsonize)

    logger = logging.getLogger("dspy_monolithic")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_split()
    if args.limit:
        cases = cases[:args.limit]

    ledger = []
    for case in cases:
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        try:
            result = pipe(report=report, logger=logger, fname=case["id"])
        except Exception as e:
            logger.error(f"{case['id']}: {e}")
            result = {"_error": str(e)}
        elapsed = time.perf_counter() - t0
        with (out_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ledger.append({"id": case["id"], "elapsed_s": elapsed})
        print(f"  [{case['id']}] {elapsed:.1f}s")

    with (out_dir / "_ledger.json").open("w", encoding="utf-8") as f:
        json.dump({"model": args.model, "skip_jsonize": args.skip_jsonize,
                   "runs": ledger}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
