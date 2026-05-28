"""Run Control 2: guided Phase 1 output with the original KCD retriever."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.ablation.runner import run_phase2_and_phase3


def main():
    parser = argparse.ArgumentParser(
        description="Run CLEVER Control 2: standard guided Phase 1 output + original KCD retrieval + Phase 3 correction."
    )
    parser.add_argument("--phase1-output", required=True, help="Standard CLEVER Phase 1 JSON output path.")
    parser.add_argument("--faiss-index", required=True, help="Original KCD SapBERT FAISS index directory.")
    parser.add_argument("--rules-csv", required=True, help="Rule CSV path.")
    parser.add_argument("--phase2-output", required=True, help="Path to save Control 2 Phase 2 JSON output.")
    parser.add_argument("--phase3-output", required=True, help="Path to save Control 2 Phase 3 JSON output.")
    parser.add_argument("--rule-log", required=True, help="Path to save applied-rule log CSV.")
    args = parser.parse_args()

    run_phase2_and_phase3(
        phase1_output=args.phase1_output,
        faiss_index=args.faiss_index,
        rules_csv=args.rules_csv,
        phase2_output=args.phase2_output,
        phase3_output=args.phase3_output,
        rule_log=args.rule_log,
        phase1_schema="guided",
    )


if __name__ == "__main__":
    main()

