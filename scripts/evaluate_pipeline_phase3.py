from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.evaluation.phase3_metrics import run_phase3_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run the original cumulative CLEVER Phase 3 code evaluation.")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON path.")
    parser.add_argument("--phase3-output", required=True, help="Phase 3 JSON output path.")
    parser.add_argument("--output-excel", required=True, help="Path to save Phase 3 evaluation XLSX.")
    args = parser.parse_args()

    run_phase3_evaluation(
        gt_path=args.ground_truth,
        model_output_path=args.phase3_output,
        output_xlsx=args.output_excel,
    )


if __name__ == "__main__":
    main()
