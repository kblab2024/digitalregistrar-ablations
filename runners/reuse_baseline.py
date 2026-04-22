"""
Cell A — reuse the existing modular-DSPy outputs.

The modular DSPy × gpt-oss:20b run IS the parent project's pipeline;
the authors should already have predictions for the test split in some
directory (for example, wherever `digitalregistrar/experiment.py`
writes). This script simply copies those per-case JSONs into
`results/dspy_modular_<model>/` so every cell has a consistent
on-disk layout for the aggregator.

We accept an arbitrary source directory for each model — the script
filters by filename to only copy cases present in the test split.

Usage:
    python runners/reuse_baseline.py \\
        --modular-gpt-oss-dir E:/experiment/20260422/gpt-oss \\
        --modular-gpt4-dir  ../digitalregistrar-benchmarks/results/gpt4_dspy
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT.parent / "digitalregistrar-benchmarks" / "data" / "splits.json"


def _case_ids() -> list[str]:
    with SPLITS_PATH.open(encoding="utf-8") as f:
        return [c["id"] for c in json.load(f)["test"]]


def _copy_matching(src_dir: Path, dst_dir: Path, case_ids: list[str]) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid in case_ids:
        # Accept either `<id>.json` or `<id>_output.json` conventions.
        for candidate in (src_dir / f"{cid}.json",
                          src_dir / f"{cid}_output.json"):
            if candidate.exists():
                shutil.copy2(candidate, dst_dir / f"{cid}.json")
                n += 1
                break
        else:
            print(f"  [miss] {cid} not found in {src_dir}", file=sys.stderr)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modular-gpt-oss-dir", type=Path, default=None,
                    help="directory with Cell A × gpt-oss predictions")
    ap.add_argument("--modular-gpt4-dir", type=Path, default=None,
                    help="directory with Cell A × gpt-4-turbo predictions")
    args = ap.parse_args()

    case_ids = _case_ids()
    results_root = ROOT / "results"

    if args.modular_gpt_oss_dir:
        n = _copy_matching(args.modular_gpt_oss_dir,
                           results_root / "dspy_modular_gpt-oss",
                           case_ids)
        print(f"Copied {n} gpt-oss predictions")

    if args.modular_gpt4_dir:
        n = _copy_matching(args.modular_gpt4_dir,
                           results_root / "dspy_modular_gpt4",
                           case_ids)
        print(f"Copied {n} gpt-4-turbo predictions")


if __name__ == "__main__":
    main()
