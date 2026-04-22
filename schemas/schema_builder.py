"""
Loads the per-organ JSON schemas shipped in
`../digitalregistrar-annotation/schemas/*.json` and stitches them into
the prompt for the raw-JSON runner (Cell C).

Two-step approach:
    1. `load_organ_schema(organ)` returns the full nested JSON-Schema
       dict for one organ (the original file already merges all
       per-subsection signatures — e.g. breast.json includes
       BreastCancerNonnested, DCIS, BreastCancerStaging, ... under its
       top-level `properties`).
    2. `flatten_schema_for_prompt(schema)` strips it to the form the
       parent pipeline actually emits at runtime: a flat `cancer_data`
       dict produced by `.update()`-ing each subsection's output into
       a single top-level object. This is what we ask the raw LLM to
       produce so the comparison is apples-to-apples.

We also expose `validate_cancer_data(organ, d)` for Pydantic-style
post-validation on the LLM's raw output.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None  # optional; only needed for validate_cancer_data

SCHEMA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "digitalregistrar-annotation" / "schemas"
)

# Top-level JSON schema keys we expect inside a per-organ schema file.
# These are the DSPy signature class names.
SUBSECTION_KEYS_FALLBACK = {
    "breast": ["BreastCancerNonnested", "DCIS", "BreastCancerGrading",
               "BreastCancerStaging", "BreastCancerMargins",
               "BreastCancerLN", "BreastCancerBiomarkers"],
    "lung": ["LungCancerNonnested", "LungCancerStaging",
             "LungCancerMargins", "LungCancerLN",
             "LungCancerBiomarkers", "LungCancerOthernested"],
    # The loader auto-detects from the file when possible; this dict is
    # only used for error reporting and prompt field ordering.
}


@lru_cache(maxsize=None)
def load_organ_schema(organ: str) -> dict:
    """Return the raw JSON-Schema dict for the organ."""
    path = SCHEMA_ROOT / f"{organ}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No JSON schema at {path}. Available schemas: "
            f"{sorted(p.stem for p in SCHEMA_ROOT.glob('*.json'))}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_schema_for_prompt(schema: dict) -> dict:
    """Return a schema describing the flat cancer_data dict the parent
    pipeline emits at runtime.

    The input schema has shape:
        { properties: { BreastCancerNonnested: {properties: {...fields...}},
                        BreastCancerStaging:   {properties: {...}},
                        ... } }

    The runtime output has shape:
        { ...fields_from_nonnested..., ...fields_from_staging..., ... }
    (i.e. `.update()`-merged).
    """
    merged_props: dict = {}
    subsection_props = schema.get("properties", {})
    for subsection_name, subsection_schema in subsection_props.items():
        if not isinstance(subsection_schema, dict):
            continue
        fields = subsection_schema.get("properties", {})
        for field_name, field_schema in fields.items():
            if field_name in merged_props:
                # First-wins, matching CancerPipeline.forward semantics.
                continue
            merged_props[field_name] = field_schema

    return {
        "type": "object",
        "title": schema.get("title", "cancer_data"),
        "properties": merged_props,
        "additionalProperties": False,
    }


def describe_field_list(flat_schema: dict) -> str:
    """Return a human-readable field checklist for the prompt. We keep
    it concise: `  - field_name (type, optional): description`."""
    lines: list[str] = []
    for name, spec in flat_schema.get("properties", {}).items():
        type_desc = _spec_type_label(spec)
        desc = spec.get("description", "")
        lines.append(f"  - {name} ({type_desc}): {desc}")
    return "\n".join(lines)


def _spec_type_label(spec: dict) -> str:
    if "anyOf" in spec:
        inner = [_spec_type_label(x) for x in spec["anyOf"]]
        return " | ".join(inner)
    if "enum" in spec:
        return f"enum{tuple(spec['enum'])}"
    if "type" in spec:
        t = spec["type"]
        if t == "array":
            return f"array<{_spec_type_label(spec.get('items', {}))}>"
        return str(t)
    if "$ref" in spec:
        return f"ref:{spec['$ref'].split('/')[-1]}"
    return "any"


def validate_cancer_data(organ: str, cancer_data: dict) -> list[str]:
    """Return a list of validation-error strings; empty if valid.
    Requires `jsonschema` to be installed."""
    if jsonschema is None:
        return ["jsonschema not installed — skipping validation"]
    schema = flatten_schema_for_prompt(load_organ_schema(organ))
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(cancer_data), key=lambda e: e.path):
        errors.append(f"{list(err.path)}: {err.message}")
    return errors


if __name__ == "__main__":
    # Smoke test: print per-organ field counts.
    for path in sorted(SCHEMA_ROOT.glob("*.json")):
        if path.stem == "common":
            continue
        schema = load_organ_schema(path.stem)
        flat = flatten_schema_for_prompt(schema)
        n = len(flat["properties"])
        print(f"{path.stem:12s}  {n:3d} flat fields")
