"""Run the direct raw-note-to-final-KCD LLM baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.direct_baseline.runner import run_direct_baseline_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-pass direct LLM baseline: raw ED note -> final KCD-8 codes. "
            "This path does not use CLEVER Phase 1, vector retrieval, or Phase 3 rules."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-workers", default=1, type=int)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--batch-dir", default=None, type=Path)
    parser.add_argument("--audit-output", default=None, type=Path)
    parser.add_argument("--manifest-output", default=None, type=Path)
    parser.add_argument("--max-records", default=None, type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="Store raw model responses in the audit file. Off by default to reduce PHI leakage risk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = run_direct_baseline_dir(
        input_dir=args.input_dir,
        output_json=args.output,
        model_id=args.model_id,
        region_name=args.region,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        batch_dir=args.batch_dir,
        audit_json=args.audit_output,
        manifest_json=args.manifest_output,
        resume=not args.no_resume,
        save_raw_responses=args.save_raw_responses,
        max_records=args.max_records,
    )
    print(f"Saved direct-baseline predictions for {len(predictions)} records to {args.output}")


if __name__ == "__main__":
    main()
