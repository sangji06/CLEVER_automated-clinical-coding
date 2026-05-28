from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.phase2_code_retriever.pipeline import run_phase2_retrieval_only
from clever.phase3_code_corrector.rule_apply import KCDRuleProcessor


def main():
    parser = argparse.ArgumentParser(description="Run public CLEVER Phase 2 -> Phase 3 using existing Phase 1 output.")
    parser.add_argument("--faiss-index", required=True, help="Path to the SapBERT FAISS index directory.")
    parser.add_argument("--phase1-output", required=True, help="Phase 1 JSON output path.")
    parser.add_argument("--rules-csv", required=True, help="Rule CSV path.")
    parser.add_argument("--phase2-output", required=True, help="Path to save Phase 2 JSON output.")
    parser.add_argument("--phase3-output", required=True, help="Path to save Phase 3 JSON output.")
    parser.add_argument("--rule-log", required=True, help="Path to save Phase 3 applied-rule log CSV.")
    args = parser.parse_args()

    run_phase2_retrieval_only(
        faiss_index_path=args.faiss_index,
        model_output_path=args.phase1_output,
        json_output_path=args.phase2_output,
    )

    processor = KCDRuleProcessor()
    if processor.load_rules(args.rules_csv):
        processor.process_json(
            input_json_path=args.phase2_output,
            output_json_path=args.phase3_output,
            log_csv_path=args.rule_log,
        )


if __name__ == "__main__":
    main()
