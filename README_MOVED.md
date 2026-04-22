# digitalregistrar-ablations (deprecated)

The contents of this folder have been folded into the unified research project:

> **`d:/localcode/digital-registrar-research/`** — see `src/digital_registrar_research/ablations/`

For research work, use the new package. This folder is left in place for one release cycle and may be deleted afterwards.

The original `README.md` is preserved alongside this notice. Notable migration points:

- `runners/{dspy_monolithic, raw_json, reuse_baseline}.py` → `digital_registrar_research/ablations/runners/...`
- `signatures/monolithic.py` → `digital_registrar_research/ablations/signatures/monolithic.py` (no more `sys.path.insert` to parent)
- `eval/run_ablations.py` → `digital_registrar_research/ablations/eval/run_ablations.py` (imports canonical scoring from `digital_registrar_research.benchmarks.eval`)
- `schemas/schema_builder.py` → `digital_registrar_research/schemas/builder.py` (and now reads from the canonical `schemas/data/` rather than the annotation folder)
