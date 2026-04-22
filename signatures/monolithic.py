"""
Monolithic DSPy signatures — one per organ, collapsed from the 5–7
per-subsection signatures that ship with the parent project.

Rather than hand-copy every field, we introspect the existing organ
signatures (BreastCancerNonnested, BreastCancerStaging, ...) and merge
their output fields into a single new `dspy.Signature` subclass per
organ. This keeps the monolithic baseline automatically in sync with
the modular baseline as the parent project evolves.

Usage:
    from signatures.monolithic import get_monolithic_signature
    sig = get_monolithic_signature("breast")
    predictor = dspy.Predict(sig)
    result = predictor(report=paragraphs, report_jsonized={})
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Import the parent project without modifying it.
PARENT = Path(__file__).resolve().parents[2] / "digitalregistrar"
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import dspy  # noqa: E402
from models.modellist import organmodels  # noqa: E402

# Wildcard import every organ module so all signature classes are in
# this namespace — mirrors how pipeline.py resolves them.
from models.common import *       # noqa: E402, F401, F403
from models.breast import *       # noqa: E402, F401, F403
from models.lung import *         # noqa: E402, F401, F403
from models.colon import *        # noqa: E402, F401, F403
from models.prostate import *     # noqa: E402, F401, F403
from models.esophagus import *    # noqa: E402, F401, F403
from models.pancreas import *     # noqa: E402, F401, F403
from models.thyroid import *      # noqa: E402, F401, F403
from models.cervix import *       # noqa: E402, F401, F403
from models.liver import *        # noqa: E402, F401, F403
from models.stomach import *      # noqa: E402, F401, F403


MONOLITHIC_DOCSTRING = (
    "You are a cancer registrar. Extract ALL structured fields listed "
    "below from the given {organ} cancer excision report in a single "
    "pass. DO NOT JUST RETURN NULL. If an item is not present in the "
    "report, return null for that item, but try your best to fill in "
    "the others. Return every field in your response even if it is "
    "null — do not omit fields."
)

INPUT_FIELD_NAMES = {"report", "report_jsonized"}


def _iter_output_fields(cls: type) -> list[tuple[str, object, object]]:
    """Yield (name, type_hint, dspy.OutputField descriptor) for each
    output field declared on a DSPy signature class."""
    out = []
    annotations = getattr(cls, "__annotations__", {})
    for name, type_hint in annotations.items():
        if name in INPUT_FIELD_NAMES:
            continue
        descriptor = cls.__dict__.get(name)
        if descriptor is None:
            continue
        out.append((name, type_hint, descriptor))
    return out


@lru_cache(maxsize=None)
def get_monolithic_signature(organ: str) -> type[dspy.Signature]:
    """Return a dynamically-built dspy.Signature class that contains
    every output field declared by the per-subsection signatures for
    the given organ.

    Field conflicts (same name declared by multiple subsection
    signatures) are resolved by first-wins — which matches the parent
    pipeline's `output_report["cancer_data"].update(organ_data)`
    ordering in `pipeline.CancerPipeline.forward`.
    """
    if organ not in organmodels:
        raise ValueError(f"Unknown organ '{organ}'. "
                         f"Known: {sorted(organmodels)}")

    merged_annotations: dict[str, object] = {}
    merged_attrs: dict[str, object] = {}
    seen_fields: set[str] = set()

    for sig_name in organmodels[organ]:
        cls = globals().get(sig_name)
        if cls is None:
            raise RuntimeError(
                f"Per-subsection signature {sig_name!r} not importable — "
                f"check the wildcard imports at the top of this module.")
        for name, type_hint, descriptor in _iter_output_fields(cls):
            if name in seen_fields:
                continue
            seen_fields.add(name)
            merged_annotations[name] = type_hint
            merged_attrs[name] = descriptor

    # Two standard input fields, identical to the parent modular design.
    merged_annotations = {
        "report": list,
        "report_jsonized": dict,
        **merged_annotations,
    }
    merged_attrs["report"] = dspy.InputField(
        desc="The pathological report for this cancer excision, "
             "separated into paragraphs.")
    merged_attrs["report_jsonized"] = dspy.InputField(
        desc="A roughly structured JSON summary of the report, produced "
             "by an upstream signature. May be an empty dict if that step "
             "is skipped.")
    merged_attrs["__annotations__"] = merged_annotations
    merged_attrs["__doc__"] = MONOLITHIC_DOCSTRING.format(organ=organ)

    cls_name = f"{organ.title()}CancerMonolithic"
    sig_cls = type(cls_name, (dspy.Signature,), merged_attrs)
    return sig_cls


def list_monolithic_fields(organ: str) -> list[str]:
    """Return the ordered list of output field names that the monolithic
    signature for this organ will produce. Useful for validation and
    for driving prompt construction in the raw-JSON runner."""
    sig = get_monolithic_signature(organ)
    return [
        name for name in sig.__annotations__
        if name not in INPUT_FIELD_NAMES
    ]


if __name__ == "__main__":
    # Smoke test: list field counts per organ.
    for organ in sorted(organmodels):
        fields = list_monolithic_fields(organ)
        print(f"{organ:12s}  {len(fields):3d} fields")
