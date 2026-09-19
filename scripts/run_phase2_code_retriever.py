from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.phase2_code_retriever.pipeline import run_phase2_retrieval_only


def main():
    parser = argparse.ArgumentParser(description="Run CLEVER Phase 2 (Code Retriever) without GT evaluation.")
    parser.add_argument("--faiss-index", required=True, help="Path to the SapBERT FAISS index directory.")
    parser.add_argument("--phase1-output", required=True, help="Phase 1 JSON output path.")
    parser.add_argument("--output-json", required=True, help="Path to save Phase 2 JSON output.")
    args = parser.parse_args()

    run_phase2_retrieval_only(
        faiss_index_path=args.faiss_index,
        model_output_path=args.phase1_output,
        json_output_path=args.output_json,
    )


if __name__ == "__main__":
    main()
