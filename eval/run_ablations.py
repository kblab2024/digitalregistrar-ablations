"""
Aggregate ablation cell predictions into a single grid.

Reuses the scoring harness at
`../digitalregistrar-benchmarks/eval/metrics.py` so the definitions of
"field correct", "attempted", and the nested bipartite match are the
same as in the main comparison tables.

Output files, all under `../results/`:

    ablation_grid.csv         long-form: one row per (cell, model, case, field)
    ablation_summary.csv      per-(cell, model, field): accuracy + coverage
    ablation_table.csv        pivot: rows=field, cols=<cell>_<model>, cells=accuracy
    cell_deltas.csv           A-vs-B and B-vs-C per-field deltas per model
    efficiency.csv            mean latency + schema-error rate per cell/model

Usage:
    python eval/run_ablations.py
    python eval/run_ablations.py --cells dspy_modular dspy_monolithic raw_json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT.parent / "digitalregistrar-benchmarks"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from eval.metrics import (  # noqa: E402  (this is the benchmarks' eval module)
    aggregate_to_csv as _bench_aggregate,
    summary_table,
)

RESULTS = ROOT / "results"
SPLITS_PATH = BENCH_ROOT / "data" / "splits.json"
GOLD_ROOT = Path(r"d:/localcode/example_folders/tcga_annotation_20251117")

DEFAULT_CELLS = ["dspy_modular", "dspy_monolithic", "raw_json"]
DEFAULT_MODELS = ["gpt-oss", "gpt4"]


def _method_key(cell: str, model: str) -> str:
    return f"{cell}_{model}"


def _discover(cells: list[str], models: list[str]) -> dict[str, Path]:
    """Build the `method_to_preds` dict expected by benchmarks/eval/metrics.py.
    Keys are `<cell>_<model>`; missing folders are dropped with a warning."""
    method_to_preds: dict[str, Path] = {}
    for cell in cells:
        for model in models:
            p = RESULTS / f"{cell}_{model}"
            if p.exists() and any(p.glob("*.json")):
                method_to_preds[_method_key(cell, model)] = p
            else:
                print(f"[warn] no predictions at {p} — skipping")
    return method_to_preds


def compute_efficiency(cells: list[str], models: list[str]) -> pd.DataFrame:
    """Pull timings + error counts from each cell's `_ledger.json`."""
    rows = []
    for cell in cells:
        for model in models:
            ledger = RESULTS / f"{cell}_{model}" / "_ledger.json"
            if not ledger.exists():
                continue
            with ledger.open(encoding="utf-8") as f:
                data = json.load(f)
            runs = data.get("runs", [])
            latencies = [r["elapsed_s"] for r in runs
                         if isinstance(r.get("elapsed_s"), (int, float))]
            schema_errors = 0
            parse_errors = 0
            for path in (RESULTS / f"{cell}_{model}").glob("*.json"):
                if path.name.startswith("_"):
                    continue
                with path.open(encoding="utf-8") as f:
                    try:
                        pred = json.load(f)
                    except Exception:
                        parse_errors += 1
                        continue
                if pred.get("_schema_errors"):
                    schema_errors += 1
                if pred.get("_parse_error") or pred.get("_error"):
                    parse_errors += 1
            rows.append({
                "cell": cell,
                "model": model,
                "n_cases": len(runs),
                "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
                "median_latency_s": sorted(latencies)[len(latencies) // 2]
                                     if latencies else None,
                "schema_errors": schema_errors,
                "parse_errors": parse_errors,
                "validation_retries": data.get("validation_retries", 0),
            })
    return pd.DataFrame(rows)


def compute_cell_deltas(long_df: pd.DataFrame,
                        cells: list[str],
                        models: list[str]) -> pd.DataFrame:
    """For each (model, field): A→B and B→C accuracy delta.

    Positive A→B delta = monolithic beats modular = modularity did NOT help.
    Negative = modularity helped.
    """
    rows = []
    for model in models:
        # Filter to numeric accuracy rows (drop Nones / nested floats).
        df = long_df.copy()
        df = df[df["attempted"] == True]  # noqa: E712
        df["accuracy"] = pd.to_numeric(df["correct"], errors="coerce")

        def mean_accuracy(cell_name: str, field: str) -> float | None:
            sub = df[(df["method"] == _method_key(cell_name, model))
                     & (df["field"] == field)]
            if sub.empty:
                return None
            return float(sub["accuracy"].mean())

        fields = sorted(df["field"].unique())
        for field in fields:
            a = mean_accuracy(cells[0], field) if len(cells) > 0 else None
            b = mean_accuracy(cells[1], field) if len(cells) > 1 else None
            c = mean_accuracy(cells[2], field) if len(cells) > 2 else None
            rows.append({
                "model": model, "field": field,
                f"{cells[0]}": a,
                f"{cells[1]}": b,
                f"{cells[2]}" if len(cells) > 2 else None: c,
                "modularity_effect_A_minus_B":
                    (a - b) if (a is not None and b is not None) else None,
                "framework_effect_B_minus_C":
                    (b - c) if (b is not None and c is not None) else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    method_to_preds = _discover(args.cells, args.models)
    if not method_to_preds:
        sys.exit("No prediction folders found under results/. "
                 "Run the cell runners first.")

    grid_csv = RESULTS / "ablation_grid.csv"
    long_df = _bench_aggregate(method_to_preds, GOLD_ROOT, SPLITS_PATH, grid_csv)
    print(f"Wrote {grid_csv}  ({len(long_df)} rows)")

    summary = summary_table(long_df)
    summary_path = RESULTS / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Publication-ready pivot: per-field × (cell, model).
    pivot = summary.pivot_table(
        index="field", columns="method",
        values="accuracy_attempted", aggfunc="first",
    )
    pivot.to_csv(RESULTS / "ablation_table.csv")
    print(f"Wrote {RESULTS / 'ablation_table.csv'}")

    deltas = compute_cell_deltas(long_df, args.cells, args.models)
    deltas.to_csv(RESULTS / "cell_deltas.csv", index=False)
    print(f"Wrote {RESULTS / 'cell_deltas.csv'}")

    eff = compute_efficiency(args.cells, args.models)
    eff.to_csv(RESULTS / "efficiency.csv", index=False)
    print(f"Wrote {RESULTS / 'efficiency.csv'}")

    print("\nper-method mean accuracy:")
    print(pivot.mean().to_string())


if __name__ == "__main__":
    main()
