#!/usr/bin/env python3
"""Calculate Reviewer 4 source-stratified recall for Phase 1 Module 1.1.

This analysis compares the same endpoint used for Table 1 Module 1.1:

* reference: unique normalized terms in ``tab1_annotations``;
* prediction: unique normalized terms in ``step3_filtering``;
* matching: document-level exact normalized text matching.

The script does not read raw notes. It reuses field assignments already saved
in the refined-term provenance CSV, linking them back to tab1 annotations by
document, character span, and entity text.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIAGNOSIS_FIELDS = {
    "임상진단명 자동완성검색기능#29",
    "진단명-Free Text#30",
}
DIAGNOSIS_FIELD_ONLY = "DIAGNOSIS_FIELD_ONLY"
FIELD_PLUS_NARRATIVE = "FIELD_PLUS_NARRATIVE"
NARRATIVE_ONLY = "NARRATIVE_ONLY"
SOURCE_GROUPS = (
    DIAGNOSIS_FIELD_ONLY,
    FIELD_PLUS_NARRATIVE,
    NARRATIVE_ONLY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--refined-source-csv", required=True, type=Path)
    parser.add_argument(
        "--manual-source-overrides-csv",
        type=Path,
        help=(
            "Optional private CSV for annotations without a reconstructed field. "
            "Columns: Document_ID, GT_Term, Source_Group, Review_Note."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=step1_result.json; repeat once per model",
    )
    parser.add_argument("--table1-metrics-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--expected-document-count", type=int, default=300)
    return parser.parse_args()


def normalize_id(value: Any) -> str:
    text = str(value or "")
    return text[:-4] if text.endswith(".txt") else text


def normalize_term(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return "".join(text.lower().split())


def document_stratum(document_id: str) -> str:
    parts = document_id.split("_")
    return parts[1] if len(parts) >= 3 else "Unknown"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_gt(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        document_id = normalize_id(record.get("document_id"))
        if not document_id:
            continue
        sequence = int(document_id.split("_", 1)[0])
        if sequence > 50:
            selected[document_id] = record
    return selected


def load_refined_source_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    by_document: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            by_document[normalize_id(row.get("Document_ID"))].append(row)
    return dict(by_document)


def load_manual_source_overrides(
    path: Path | None,
) -> dict[tuple[str, str], tuple[str, str]]:
    if path is None:
        return {}
    overrides: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            document_id = normalize_id(row.get("Document_ID"))
            term = normalize_term(row.get("GT_Term"))
            group = str(row.get("Source_Group") or "").strip()
            if not document_id or not term or group not in SOURCE_GROUPS:
                raise ValueError(f"Invalid manual source override row: {row}")
            key = (document_id, term)
            if key in overrides:
                raise ValueError(f"Duplicate manual source override: {key}")
            overrides[key] = (group, str(row.get("Review_Note") or "").strip())
    return overrides


def build_span_field_map(
    gt: dict[str, dict[str, Any]],
    source_rows: dict[str, list[dict[str, str]]],
) -> dict[str, dict[tuple[Any, Any, str], str]]:
    mapping: dict[str, dict[tuple[Any, Any, str], str]] = defaultdict(dict)
    for document_id, record in gt.items():
        terms = record.get("tab2_refined_terms") or []
        rows = source_rows.get(document_id) or []
        if len(terms) != len(rows):
            raise ValueError(
                f"Refined GT/source row count differs for {document_id}: "
                f"{len(terms)} vs {len(rows)}"
            )
        for term_object, row in zip(terms, rows):
            if str(term_object.get("term") or "") != str(row.get("GT_Term") or ""):
                raise ValueError(f"Refined GT/source row order differs for {document_id}")
            entities = term_object.get("original_entities") or []
            fields = (
                str(row.get("Detected_Fields") or "").split(" | ")
                if row.get("Detected_Fields")
                else []
            )
            if len(entities) != len(fields):
                raise ValueError(
                    f"Entity/field count differs for {document_id} / {term_object.get('term')}"
                )
            for entity, field in zip(entities, fields):
                key = (
                    entity.get("start"),
                    entity.get("end"),
                    normalize_term(entity.get("entity") or entity.get("term")),
                )
                previous = mapping[document_id].get(key)
                if previous is not None and previous != field:
                    raise ValueError(f"Conflicting field assignment for {document_id} / {key}")
                mapping[document_id][key] = field
    return {document_id: dict(values) for document_id, values in mapping.items()}


def classify_tab1_terms(
    gt: dict[str, dict[str, Any]],
    span_fields: dict[str, dict[tuple[Any, Any, str], str]],
    manual_source_overrides: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    manual_source_overrides = manual_source_overrides or {}
    output_rows: list[dict[str, Any]] = []
    source_by_document_term: dict[str, dict[str, str]] = defaultdict(dict)
    unresolved: list[tuple[str, str]] = []

    for document_id in sorted(gt):
        occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in gt[document_id].get("tab1_annotations") or []:
            term = normalize_term(annotation.get("entity") or annotation.get("term"))
            if term:
                occurrences[term].append(annotation)

        for term, annotations in sorted(occurrences.items()):
            fields: list[str] = []
            missing_occurrences = 0
            displayed_terms: list[str] = []
            for annotation in annotations:
                displayed_terms.append(str(annotation.get("entity") or annotation.get("term") or ""))
                key = (
                    annotation.get("start"),
                    annotation.get("end"),
                    term,
                )
                field = span_fields.get(document_id, {}).get(key)
                if field is None:
                    missing_occurrences += 1
                else:
                    fields.append(field)

            override = manual_source_overrides.get((document_id, term))
            review_status = "AUTO_RECONSTRUCTED"
            review_note = ""
            if missing_occurrences:
                if override is None:
                    unresolved.append((document_id, displayed_terms[0]))
                    continue
                override_group, review_note = override
                review_status = "USER_CONFIRMED_SOURCE"
            else:
                override_group = ""

            has_diagnosis_field = any(field in DIAGNOSIS_FIELDS for field in fields)
            has_narrative = any(field not in DIAGNOSIS_FIELDS for field in fields)
            if override_group:
                if override_group == NARRATIVE_ONLY:
                    has_narrative = True
                elif override_group == DIAGNOSIS_FIELD_ONLY:
                    has_diagnosis_field = True
                else:
                    has_diagnosis_field = True
                    has_narrative = True

            if has_diagnosis_field and has_narrative:
                group = FIELD_PLUS_NARRATIVE
            elif has_diagnosis_field:
                group = DIAGNOSIS_FIELD_ONLY
            elif has_narrative:
                group = NARRATIVE_ONLY
            else:
                unresolved.append((document_id, displayed_terms[0]))
                continue

            source_by_document_term[document_id][term] = group
            output_rows.append(
                {
                    "Document_ID": document_id,
                    "GT_Term": displayed_terms[0],
                    "Normalized_GT_Term": term,
                    "Occurrence_Count": len(annotations),
                    "Detected_Fields": " | ".join(sorted(set(fields))),
                    "Source_Group": group,
                    "Review_Status": review_status,
                    "Review_Note": review_note,
                }
            )

    if unresolved:
        raise ValueError(f"Unresolved tab1 source assignments: {unresolved}")
    return output_rows, {key: dict(value) for key, value in source_by_document_term.items()}


def parse_model(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Invalid model specification: {specification}")
    name, path = specification.split("=", 1)
    return name.strip(), Path(path)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {normalize_id(key): value for key, value in payload.items()}


def prediction_terms(record: dict[str, Any]) -> set[str]:
    terms = {
        normalize_term(item.get("entity") or item.get("term"))
        for item in (record.get("step3_filtering") or [])
        if isinstance(item, dict)
    }
    terms.discard("")
    return terms


def bootstrap_recall(
    per_document: dict[str, dict[str, tuple[int, int]]],
    group: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    strata: dict[str, list[str]] = defaultdict(list)
    for document_id in per_document:
        strata[document_stratum(document_id)].append(document_id)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        recovered = 0
        total = 0
        for documents in strata.values():
            for _ in range(len(documents)):
                sampled = rng.choice(documents)
                recovered += per_document[sampled][group][0]
                total += per_document[sampled][group][1]
        estimates.append(recovered / total if total else 1.0)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_table1_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["model"]: row
        for row in rows
        if row.get("endpoint") == "Phase 1 Module 1.1 entity extraction"
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gt = load_gt(args.ground_truth)
    if len(gt) != args.expected_document_count:
        raise ValueError(f"Expected {args.expected_document_count} documents, found {len(gt)}")
    strata = Counter(document_stratum(document_id) for document_id in gt)
    if strata != Counter({"Very-Long": 75, "Long": 75, "Medium": 75, "Short": 75}):
        raise ValueError(f"Unexpected length strata: {strata}")

    refined_rows = load_refined_source_rows(args.refined_source_csv)
    span_fields = build_span_field_map(gt, refined_rows)
    manual_source_overrides = load_manual_source_overrides(
        args.manual_source_overrides_csv
    )
    review_rows, source_by_document_term = classify_tab1_terms(
        gt,
        span_fields,
        manual_source_overrides,
    )
    if len(review_rows) != sum(len(values) for values in source_by_document_term.values()):
        raise ValueError("Duplicate tab1 document-term rows detected")

    table1 = load_table1_metrics(args.table1_metrics_csv)
    composition_counts = Counter(row["Source_Group"] for row in review_rows)
    composition_rows = [
        {
            "Source_Group": group,
            "GT_Terms": composition_counts[group],
            "GT_Term_Percent": composition_counts[group] / len(review_rows),
            "Documents_With_Group": len(
                {row["Document_ID"] for row in review_rows if row["Source_Group"] == group}
            ),
        }
        for group in SOURCE_GROUPS
    ]

    recall_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    model_manifest: dict[str, Any] = {}
    for model_index, specification in enumerate(args.model):
        model_name, path = parse_model(specification)
        predictions = load_predictions(path)
        per_document: dict[str, dict[str, tuple[int, int]]] = {}
        overall_tp = overall_fp = overall_fn = 0
        missing_documents: list[str] = []

        for document_id in sorted(gt):
            gt_terms = set(source_by_document_term[document_id])
            if document_id not in predictions:
                missing_documents.append(document_id)
            predicted = prediction_terms(predictions.get(document_id, {}))
            matched = predicted & gt_terms
            overall_tp += len(matched)
            overall_fp += len(predicted - gt_terms)
            overall_fn += len(gt_terms - predicted)
            per_document[document_id] = {}
            for group in SOURCE_GROUPS:
                group_gt = {
                    term
                    for term, source_group in source_by_document_term[document_id].items()
                    if source_group == group
                }
                per_document[document_id][group] = (
                    len(matched & group_gt),
                    len(group_gt),
                )

        source_recovered_sum = 0
        for group_index, group in enumerate(SOURCE_GROUPS):
            recovered = sum(values[group][0] for values in per_document.values())
            total = sum(values[group][1] for values in per_document.values())
            source_recovered_sum += recovered
            ci_low, ci_high = bootstrap_recall(
                per_document,
                group,
                repetitions=args.bootstrap,
                seed=args.seed + model_index * 100 + group_index,
            )
            recall_rows.append(
                {
                    "Model": model_name,
                    "Source_Group": group,
                    "Recovered_GT_Terms": recovered,
                    "Total_GT_Terms": total,
                    "Recall": recovered / total if total else 1.0,
                    "Recall_CI_Low": ci_low,
                    "Recall_CI_High": ci_high,
                    "Missing_Prediction_Documents": len(missing_documents),
                }
            )

        precision, recall, f1 = metrics(overall_tp, overall_fp, overall_fn)
        expected = table1.get(model_name)
        if expected is None:
            raise ValueError(f"Table 1 metrics missing model: {model_name}")
        expected_tp = int(expected["tp"])
        expected_fp = int(expected["fp"])
        expected_fn = int(expected["fn"])
        verification_rows.append(
            {
                "Model": model_name,
                "Documents": len(gt),
                "Unique_GT_Terms": overall_tp + overall_fn,
                "Overall_TP": overall_tp,
                "Overall_FP": overall_fp,
                "Overall_FN": overall_fn,
                "Overall_Precision": precision,
                "Overall_Recall": recall,
                "Overall_F1": f1,
                "Sum_of_Source_Recovered_Terms": source_recovered_sum,
                "Source_TP_Equals_Overall_TP": source_recovered_sum == overall_tp,
                "Table1_TP": expected_tp,
                "Table1_FP": expected_fp,
                "Table1_FN": expected_fn,
                "Counts_Equal_Table1": (
                    (overall_tp, overall_fp, overall_fn)
                    == (expected_tp, expected_fp, expected_fn)
                ),
                "Missing_Prediction_Documents": len(missing_documents),
            }
        )
        if not verification_rows[-1]["Source_TP_Equals_Overall_TP"]:
            raise ValueError(f"Source TP does not reconcile for {model_name}")
        if not verification_rows[-1]["Counts_Equal_Table1"]:
            raise ValueError(f"Module 1.1 counts do not match Table 1 for {model_name}")
        model_manifest[model_name] = {
            "prediction_file": str(path),
            "prediction_documents_present": len(set(predictions) & set(gt)),
            "missing_prediction_document_ids": missing_documents,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase1_module11_source_review.csv", review_rows)
    write_csv(args.output_dir / "phase1_module11_source_composition.csv", composition_rows)
    write_csv(args.output_dir / "phase1_module11_source_recall_by_model.csv", recall_rows)
    write_csv(args.output_dir / "phase1_module11_table1_verification.csv", verification_rows)
    manifest = {
        "endpoint": "Phase 1 Module 1.1 entity extraction",
        "reference": "tab1_annotations",
        "prediction": "step3_filtering",
        "matching": "document-level unique NFKC/lowercase/whitespace-removed exact term matching",
        "document_count": len(gt),
        "length_strata": dict(sorted(strata.items())),
        "unique_reference_term_count": len(review_rows),
        "source_group_counts": dict(composition_counts),
        "user_confirmed_manual_source_rows": sum(
            row["Review_Status"] == "USER_CONFIRMED_SOURCE"
            for row in review_rows
        ),
        "pending_manual_source_confirmation": 0,
        "bootstrap": {
            "unit": "document",
            "stratified_by": "note length",
            "repetitions": args.bootstrap,
            "seed": args.seed,
        },
        "source_specific_precision_or_f1_calculated": False,
        "reason": "Unmatched predictions have no reference-source category.",
        "models": model_manifest,
    }
    (args.output_dir / "phase1_module11_source_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Documents: {len(gt)}")
    print(f"Unique Module 1.1 GT terms: {len(review_rows)}")
    print(f"Source groups: {dict(composition_counts)}")
    for row in recall_rows:
        print(
            f"{row['Model']} | {row['Source_Group']} | "
            f"{row['Recovered_GT_Terms']}/{row['Total_GT_Terms']} | "
            f"R={row['Recall']:.4f} "
            f"(95% CI {row['Recall_CI_Low']:.4f}-{row['Recall_CI_High']:.4f})"
        )
    print("All overall counts exactly match the final Table 1 Module 1.1 analysis.")


if __name__ == "__main__":
    main()
