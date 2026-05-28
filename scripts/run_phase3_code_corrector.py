from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.phase3_code_corrector.rule_apply import KCDRuleProcessor


def main():
    parser = argparse.ArgumentParser(description="Run CLEVER Phase 3 (Code Corrector).")
    parser.add_argument("--rules-csv", required=True, help="Rule CSV path, equivalent to original rule.csv.")
    parser.add_argument("--input-json", required=True, help="Phase 2 JSON output path.")
    parser.add_argument("--output-json", required=True, help="Path to save Phase 3 JSON output.")
    parser.add_argument("--log-csv", required=True, help="Path to save applied-rule log CSV.")
    args = parser.parse_args()

    processor = KCDRuleProcessor()
    if processor.load_rules(args.rules_csv):
        processor.process_json(
            input_json_path=args.input_json,
            output_json_path=args.output_json,
            log_csv_path=args.log_csv,
        )


if __name__ == "__main__":
    main()
