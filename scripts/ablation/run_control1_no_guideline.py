"""Run Control 1: no-guideline extraction with the curated KCD retriever."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.ablation.no_guideline import run_no_guideline_extraction_dir
from clever.ablation.runner import run_phase2_and_phase3


def main():
    parser = argparse.ArgumentParser(
        description="Run CLEVER Control 1: no-guideline Phase 1 + curated KCD retrieval + Phase 3 correction."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input .txt records.")
    parser.add_argument("--model-id", required=True, help="Bedrock model id for no-guideline Phase 1.")
    parser.add_argument("--faiss-index", required=True, help="Curated KCD SapBERT FAISS index directory.")
    parser.add_argument("--rules-csv", required=True, help="Rule CSV path.")
    parser.add_argument("--phase1-output", required=True, help="Path to save no-guideline Phase 1 JSON output.")
    parser.add_argument("--phase2-output", required=True, help="Path to save Control 1 Phase 2 JSON output.")
    parser.add_argument("--phase3-output", required=True, help="Path to save Control 1 Phase 3 JSON output.")
    parser.add_argument("--rule-log", required=True, help="Path to save applied-rule log CSV.")
    parser.add_argument("--region", default="us-west-2", help="AWS Bedrock region.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of concurrent Phase 1 workers.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of records per saved no-guideline batch.")
    parser.add_argument("--batch-dir", default=None, help="Directory for baseline_batch_###.json files.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing batch files.")
    args = parser.parse_args()

    run_no_guideline_extraction_dir(
        input_dir=args.input_dir,
        output_json=args.phase1_output,
        model_id=args.model_id,
        region_name=args.region,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        batch_dir=args.batch_dir,
        resume=not args.no_resume,
    )

    run_phase2_and_phase3(
        phase1_output=args.phase1_output,
        faiss_index=args.faiss_index,
        rules_csv=args.rules_csv,
        phase2_output=args.phase2_output,
        phase3_output=args.phase3_output,
        rule_log=args.rule_log,
        phase1_schema="no_guideline",
    )


if __name__ == "__main__":
    main()


