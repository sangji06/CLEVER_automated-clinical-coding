from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.evaluation.phase2_independent import run_phase2_independent_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run the original independent Phase 2 code-mapping evaluation.")
    parser.add_argument("--input-csv", required=True, help="Input CSV, equivalent to original Step2_code_list.csv.")
    parser.add_argument("--output-excel", required=True, help="Path to save evaluation XLSX.")
    args = parser.parse_args()

    run_phase2_independent_evaluation(args.input_csv, args.output_excel)


if __name__ == "__main__":
    main()
