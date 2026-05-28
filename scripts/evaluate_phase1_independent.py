"""Evaluate CLEVER Phase 1 (Guided Extractor) output against GT JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.evaluation.phase1_metrics import run_phase1_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 1 diagnosis and symptom extraction."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_phase1_evaluation(
        model_path=str(args.predictions),
        gt_path=str(args.ground_truth),
        output_xlsx=str(args.output),
    )
    print(f"Saved Phase 1 evaluation report to {args.output}")


if __name__ == "__main__":
    main()
