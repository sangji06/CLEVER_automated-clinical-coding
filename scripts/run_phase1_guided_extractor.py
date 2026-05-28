"""Run CLEVER Phase 1 (Guided Extractor) on text records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.phase1_guided_extractor.extractor import run_extraction_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 guided diagnosis and symptom extraction."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-workers", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_extraction_dir(
        input_dir=args.input_dir,
        output_json=args.output,
        model_id=args.model_id,
        region_name=args.region,
        max_workers=args.max_workers,
    )
    print(f"Saved Phase 1 output to {args.output}")


if __name__ == "__main__":
    main()
