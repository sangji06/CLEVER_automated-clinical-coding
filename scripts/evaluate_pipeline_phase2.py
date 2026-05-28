from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.evaluation.phase2_pipeline import run_phase2_pipeline_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved CLEVER Phase 2 output against ground truth.")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON path.")
    parser.add_argument("--phase2-output", required=True, help="Phase 2 JSON output path.")
    parser.add_argument("--output-excel", required=True, help="Path to save Phase 2 evaluation XLSX.")
    args = parser.parse_args()

    run_phase2_pipeline_evaluation(
        phase2_output_path=args.phase2_output,
        gt_path=args.ground_truth,
        output_excel=args.output_excel,
    )


if __name__ == "__main__":
    main()
