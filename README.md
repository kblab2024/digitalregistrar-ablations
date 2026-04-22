# digitalregistrar-ablations

Ablation study for **The Digital Registrar**. Answers two distinct
questions:

1. **Modularity axis** — does splitting each organ's extraction into
   5–7 small DSPy signatures (the published design) perform better than
   collapsing them into a single large signature per organ?
2. **Framework axis** — does DSPy itself add value, or would a raw
   LLM call with JSON-mode output and the schema inlined do just as well?

Both axes are tested against the same held-out 51-case test split used
by the main benchmarks (in `../digitalregistrar-benchmarks/`).

## Ablation grid

| Cell | Framework | Modularity | Per organ | New work? |
|---|---|---|---|---|
| **A** — `dspy_modular`   | DSPy | 5–7 small signatures | ~5–7 LM calls | baseline, reuse |
| **B** — `dspy_monolithic` | DSPy | 1 big signature per organ | 1 LM call    | yes |
| **C** — `raw_json`        | none | 1 raw call per organ, JSON mode | 1 LM call | yes |

Each cell is run against **two backbones**:

- `gpt-oss:20b` (local, via Ollama) — the paper's recommended model
- `gpt-4-turbo` (OpenAI) — frontier comparator

→ 6 total cells, 4 new runs (Cell A reuses existing pipeline outputs).

## What does each axis tell us?

- **A vs B** on the same model answers *"is the modularity worth it?"*
  If B collapses performance, the modular design is justified.
- **B vs C** on the same model answers *"is DSPy worth it once modularity
  is held constant?"* If C collapses, DSPy is doing meaningful work beyond
  being a JSON scaffold.
- **{A,B,C} × {gpt-oss:20b, gpt-4-turbo}** reveals whether the framework
  and modularity benefits interact with model capacity — a strong result
  if they do, e.g., "modularity matters more for the smaller local model."

## Layout

```
signatures/       monolithic DSPy signatures (one class per organ)
runners/          per-cell runners: dspy_monolithic, raw_json, reuse_baseline
schemas/          loader for the JSON schemas in
                  ../digitalregistrar-annotation/schemas/ (used by raw_json)
eval/             aggregation that produces the ablation grid table
results/          per-cell, per-model JSON predictions (generated)
docs/             design rationale
```

## How to reproduce

```bash
pip install -r requirements.txt

# Cell A: reuse — point at existing pipeline outputs
python runners/reuse_baseline.py \
  --modular-gpt-oss-dir /path/to/existing/gpt-oss/outputs \
  --modular-gpt4-dir    ../digitalregistrar-benchmarks/results/gpt4_dspy

# Cell B × gpt-oss:20b (local, via Ollama)
python runners/dspy_monolithic.py --model gpt --out results/dspy_monolithic_gpt-oss

# Cell B × gpt-4-turbo
OPENAI_API_KEY=... python runners/dspy_monolithic.py \
  --model gpt4 --out results/dspy_monolithic_gpt4

# Cell C × gpt-oss:20b
python runners/raw_json.py --model gpt-oss:20b \
  --api-base http://localhost:11434 --out results/raw_json_gpt-oss

# Cell C × gpt-4-turbo
OPENAI_API_KEY=... python runners/raw_json.py --model gpt-4-turbo \
  --out results/raw_json_gpt4

# Aggregate
python eval/run_ablations.py
# → results/ablation_grid.csv and results/ablation_summary.csv
```

## Reuse from sibling folders

This folder is a **sibling** of `../digitalregistrar/` (the pipeline)
and `../digitalregistrar-benchmarks/` (the 4-method comparison). Both
are imported via `sys.path` insertion; neither is modified.

- Parent pipeline: `../digitalregistrar/pipeline.py` +
  `../digitalregistrar/models/*.py` (DSPy signatures per organ).
- Benchmarks evaluator: `../digitalregistrar-benchmarks/eval/` —
  `scope.py`, `metrics.py` are reused verbatim so the scoring rules
  are identical across the two sets of tables.
- JSON schemas: `../digitalregistrar-annotation/schemas/*.json` —
  one per organ, consumed by `schemas/schema_builder.py` for the
  `raw_json` runner.

## Licence

MIT, matching the parent project.
