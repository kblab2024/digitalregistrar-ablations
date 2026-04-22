# Design rationale

This document records the reasoning behind each ablation cell and how
the results should be read. Decisions here should carry forward if the
ablation grid is extended.

## Why these three cells?

The published Digital Registrar pipeline makes two joint design
choices:

1. It splits each organ's extraction into 5–7 small DSPy signatures
   (e.g. `BreastCancerNonnested`, `BreastCancerStaging`,
   `BreastCancerMargins`, `BreastCancerLN`, `BreastCancerBiomarkers`,
   `BreastCancerGrading`, `DCIS`).
2. It uses DSPy as the LM-calling framework.

Either choice could in principle be the one doing the real work. To
disentangle them we need a 2×2:

|          | Modular                 | Monolithic                |
|----------|-------------------------|---------------------------|
| DSPy     | **A** (baseline)        | **B** (test modularity)   |
| Raw LLM  | *(not run — see below)* | **C** (test framework)    |

The modular-raw cell is skipped because (a) without DSPy's automatic
output-schema management, running N raw calls per organ with
inter-call dependency on partial outputs becomes significantly harder
to implement correctly, and (b) a clean A→B→C ladder already answers
the questions the ablation is meant to answer:

- **A vs B**: modularity effect, with DSPy held constant.
- **B vs C**: framework effect, with modularity held constant at
  monolithic.

The chain lets us attribute any A→C gap to modularity (A→B) and
framework (B→C) separately.

## Parts held constant

To isolate the modularity and framework variables, three upstream
components stay the same across all cells:

1. **`is_cancer` classifier** — the initial "is this an eligible
   cancer excision report, and which of the 10 organs?" routing step.
   Runs as the existing DSPy signature in Cells A and B, and as an
   equivalent raw JSON-mode call in Cell C. In all cases it returns
   the same flag + organ label.
2. **`ReportJsonize` step** — the intermediate "rough JSON structuring"
   signature. Kept on by default in Cells A, B (matches the baseline);
   omitted in Cell C (the monolithic raw call already has the full
   report as context). The `--skip-jsonize` flag on `dspy_monolithic.py`
   can be used to add a supplementary "no jsonize" variant of Cell B
   if we want to isolate that step too.
3. **Test split** — the 51-case stratified test split is loaded from
   `../digitalregistrar-benchmarks/data/splits.json`. Every cell
   predicts on the exact same cases.

## Schema source of truth

The raw-JSON runner in Cell C loads its per-organ JSON schemas from
`../digitalregistrar-annotation/schemas/*.json`. These schemas were
previously generated from the same DSPy signatures used by Cell A, so
any agreement between Cells A and C reflects only model / framework
behaviour, not schema drift.

## Model-framework interaction

Running each cell against both `gpt-oss:20b` and `gpt-4-turbo` lets us
see whether DSPy's scaffolding (structured output parsing, retry on
schema violation, per-field descriptions propagated into prompts) is
more valuable for the smaller local model than the frontier model.
A priori we expect:

- On `gpt-oss:20b`: **A > B ≫ C**. Modularity saves context; DSPy
  saves JSON reliability on a smaller model.
- On `gpt-4-turbo`: **A ≈ B ≈ C**. Frontier models handle both a full
  organ schema in one shot and raw JSON output reliably.

If this pattern is observed, it directly supports the paper's
core narrative that the Digital Registrar's schema-first modular
design is what makes a local LLM competitive in the first place.

## Metrics

All cells use the shared evaluator in
`../digitalregistrar-benchmarks/eval/` — same field definitions, same
bipartite match for nested lists, same coverage accounting. Headline
per-cell numbers:

- `overall_field_acc` — mean field-level accuracy over the full
  schema (not just the fair-scope subset — all cells can populate
  every field in principle).
- `structural_validity_rate` — fraction of outputs that parse as
  valid JSON matching the organ schema. We expect this to drop
  sharply for Cell C on `gpt-oss:20b`.
- `mean_tokens_per_case` — proxy for context/cost. Monolithic
  per-organ cells use ~1× the context of the raw fallback but
  ~1/5–1/7× the context of the modular pipeline (since they make one
  call rather than N). Useful for the "efficiency" axis of the paper.
- `retry_rate` — fraction of cases that required a framework-level
  retry (DSPy's automatic parse retry). Only populated for Cells A
  and B; Cell C exposes this as the manual Pydantic-validation
  retry count.

## Known risks / threats to validity

- The `gpt-oss:20b` monolithic cell may *also* be context-window
  limited for breast (≥7 nested field groups) — specifically the
  biomarkers + LN + margins combination can overflow a 16k-token
  context when combined with a long report. We detect and flag this
  in `runners/dspy_monolithic.py`; if it fires, the finding itself
  is the result (modularity is not merely *helpful* but *necessary*).
- The schemas shipped in `../digitalregistrar-annotation/schemas/`
  may drift from the DSPy signatures over time. Keep a sanity check
  in `schemas/schema_builder.py` to assert the top-level keys match
  the expected `Literal[...]` field names.
