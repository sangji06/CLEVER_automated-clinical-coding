#!/usr/bin/env python3
"""Build a term-level source-provenance review CSV for Phase 1 ground truth.

This script is intended to be run locally by the data holder.  It reads the
private notes only during local execution and writes no raw note text or field
values to the output.  The independent-test document IDs are taken from one or
more saved Phase 1 prediction JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from source_attribution import (
    SourceReconstructionError,
    classify_document_terms,
    normalize_document_id,
)


OUTPUT_COLUMNS = [
    "Document_ID",
    "GT_Term",
    "Original_Entities",
    "Original_Labels",
    "Detected_Fields",
    "Auto_Source_Group",
    "Final_Source_Group",
    "Review_Status",
    "Span_QC",
    "Review_Note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign Phase 1 GT terms to diagnosis-field-only, narrative-only, "
            "or field-plus-narrative source groups."
        )
    )
    parser.add_argument("--notes-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument(
        "--document-ids-from",
        required=True,
        type=Path,
        nargs="+",
        help="Prediction JSON file(s) whose keys define the independent test documents.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-document-count", type=int, default=300)
    return parser.parse_args()


def document_ids_from_json(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return {normalize_document_id(key) for key in payload}
    if isinstance(payload, list):
        ids = {
            normalize_document_id(item.get("document_id", ""))
            for item in payload
            if isinstance(item, dict) and item.get("document_id")
        }
        if ids:
            return ids
    raise ValueError(f"Could not extract document IDs from {path}")


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Ground truth must be a JSON list of document records.")
    records: dict[str, dict[str, Any]] = {}
    for record in payload:
        if not isinstance(record, dict) or not record.get("document_id"):
            continue
        key = normalize_document_id(record["document_id"])
        if key in records:
            raise ValueError(f"Duplicate ground-truth document ID: {key}")
        records[key] = record
    return records


def main() -> None:
    args = parse_args()
    if not args.notes_dir.is_dir():
        raise FileNotFoundError(f"Notes directory not found: {args.notes_dir}")

    selected_ids: set[str] = set()
    for path in args.document_ids_from:
        selected_ids.update(document_ids_from_json(path))

    if len(selected_ids) != args.expected_document_count:
        raise ValueError(
            f"Expected {args.expected_document_count} unique documents, "
            f"but prediction files contained {len(selected_ids)}."
        )

    gt_records = load_ground_truth(args.ground_truth)
    missing_gt = sorted(selected_ids - set(gt_records))
    if missing_gt:
        raise ValueError(
            f"Ground truth is missing {len(missing_gt)} selected documents; "
            f"first IDs: {missing_gt[:5]}"
        )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for document_id in sorted(selected_ids):
        record = gt_records[document_id]
        gt_filename = str(record.get("document_id") or f"{document_id}.txt")
        note_path = args.notes_dir / gt_filename
        if not note_path.is_file():
            fallback = args.notes_dir / f"{document_id}.txt"
            note_path = fallback if fallback.is_file() else note_path
        if not note_path.is_file():
            errors.append(f"{document_id}: note file not found")
            continue
        try:
            rows.extend(classify_document_terms(record, note_path))
        except SourceReconstructionError as error:
            errors.append(f"{document_id}: {error}")

    if errors:
        details = "\n".join(errors[:20])
        raise RuntimeError(
            f"Source reconstruction failed for {len(errors)} documents.\n{details}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    group_counts = Counter(row["Auto_Source_Group"] for row in rows)
    review_count = sum(row["Review_Status"] != "AUTO_ACCEPT" for row in rows)
    manifest = {
        "selected_document_count": len(selected_ids),
        "classified_document_count": len(selected_ids),
        "reference_term_count": len(rows),
        "auto_source_group_counts": dict(sorted(group_counts.items())),
        "human_review_required_count": review_count,
        "diagnosis_fields": [
            "임상진단명 자동완성검색기능#29",
            "진단명-Free Text#30",
        ],
        "contains_raw_note_text": False,
        "ground_truth": str(args.ground_truth),
        "document_id_sources": [str(path) for path in args.document_ids_from],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved {len(rows)} Phase 1 GT terms to {args.output}")
    print(f"Documents: {len(selected_ids)}")
    print(f"Auto source groups: {dict(sorted(group_counts.items()))}")
    print(f"Human review required: {review_count}")
    print("The output contains GT terms and field names, but no raw note text.")


if __name__ == "__main__":
    main()
