"""Assign Phase 1 reference terms to diagnosis-field or narrative sources.

The ground-truth annotation files retain character offsets for the original
entities.  The annotation application generated those offsets against a
``source_text`` made by concatenating each note-field value, with two newlines
between fields.  Reconstructing that same text lets us identify the source
field without copying raw note text into the review output.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DIAGNOSIS_FIELD_NAMES = frozenset(
    {
        "임상진단명 자동완성검색기능#29",
        "진단명-Free Text#30",
    }
)

DIAGNOSIS_FIELD_ONLY = "DIAGNOSIS_FIELD_ONLY"
NARRATIVE_ONLY = "NARRATIVE_ONLY"
FIELD_PLUS_NARRATIVE = "FIELD_PLUS_NARRATIVE"
UNRESOLVED = "UNRESOLVED"
VALID_SOURCE_GROUPS = frozenset(
    {
        DIAGNOSIS_FIELD_ONLY,
        NARRATIVE_ONLY,
        FIELD_PLUS_NARRATIVE,
        UNRESOLVED,
    }
)

SERIALIZED_FIELD_PATTERN = re.compile(r"'([^'\n]+#\d+)'\s*:\s*'")


class SourceReconstructionError(ValueError):
    """Raised when a note cannot be reconstructed as the annotation tool did."""


@dataclass(frozen=True)
class FieldSpan:
    name: str
    start: int
    end: int

    @property
    def is_diagnosis_field(self) -> bool:
        return self.name in DIAGNOSIS_FIELD_NAMES


def normalize_document_id(value: str) -> str:
    """Return a document identifier without a trailing ``.txt`` suffix."""
    value = str(value).strip()
    return value[:-4] if value.lower().endswith(".txt") else value


def normalize_entity_text(value: Any) -> str:
    """Normalize whitespace for span-content quality checks only."""
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def reconstruct_annotation_source(raw_content: str) -> tuple[str, list[FieldSpan]]:
    """Reproduce the annotation tool's value-only source text and field spans."""
    try:
        data_list = ast.literal_eval(raw_content)
    except (SyntaxError, ValueError):
        # The annotation tool fell back to the unmodified raw string when an
        # export was wrapped in outer quotes or contained literal newlines in
        # quoted field values.  In that mode, saved annotation offsets are
        # relative to ``raw_content`` itself.  Recover only field boundaries;
        # do not parse, alter, or emit the field values.
        matches = list(SERIALIZED_FIELD_PATTERN.finditer(raw_content))
        if not matches:
            raise SourceReconstructionError(
                "The note is neither a Python-like field list nor a recognized "
                "multiline pseudo-literal export."
            )
        field_spans = []
        for index, match in enumerate(matches):
            next_start = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(raw_content)
            )
            field_spans.append(
                FieldSpan(name=match.group(1), start=match.end(), end=next_start)
            )
        return raw_content, field_spans

    if not isinstance(data_list, list) or not data_list:
        raise SourceReconstructionError("The note did not contain a non-empty list.")

    source_parts: list[str] = []
    field_spans: list[FieldSpan] = []
    cursor = 0

    for item in data_list:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            value_text = str(value)
            start = cursor
            end = start + len(value_text)
            field_spans.append(FieldSpan(name=str(key), start=start, end=end))
            source_parts.append(value_text + "\n\n")
            cursor = end + 2

    if not field_spans:
        raise SourceReconstructionError("The note did not contain any named fields.")

    return "".join(source_parts).strip(), field_spans


def field_for_span(
    start: Any,
    end: Any,
    field_spans: Iterable[FieldSpan],
) -> FieldSpan | None:
    """Locate the single field that fully contains an annotation span."""
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return None
    matches = [span for span in field_spans if span.start <= start and end <= span.end]
    return matches[0] if len(matches) == 1 else None


def _source_group(fields: list[FieldSpan | None]) -> str:
    if not fields or any(field is None for field in fields):
        return UNRESOLVED
    source_types = {field.is_diagnosis_field for field in fields if field is not None}
    if source_types == {True}:
        return DIAGNOSIS_FIELD_ONLY
    if source_types == {False}:
        return NARRATIVE_ONLY
    if source_types == {False, True}:
        return FIELD_PLUS_NARRATIVE
    return UNRESOLVED


def classify_refined_term(
    document_id: str,
    term_object: dict[str, Any],
    source_text: str,
    field_spans: list[FieldSpan],
) -> dict[str, Any]:
    """Create one privacy-preserving source-review row for a Phase 1 GT term."""
    original_entities = term_object.get("original_entities") or []
    located_fields: list[FieldSpan | None] = []
    entity_texts: list[str] = []
    entity_labels: list[str] = []
    span_qc: list[str] = []

    for entity in original_entities:
        entity_text = str(entity.get("entity") or entity.get("term") or "")
        entity_texts.append(entity_text)
        entity_labels.append(str(entity.get("label") or ""))
        start = entity.get("start")
        end = entity.get("end")
        field = field_for_span(start, end, field_spans)
        located_fields.append(field)

        if field is None:
            span_qc.append("UNMAPPED_SPAN")
            continue
        observed = source_text[start:end]
        if normalize_entity_text(observed) == normalize_entity_text(entity_text):
            span_qc.append("OK")
        else:
            span_qc.append("TEXT_MISMATCH")

    auto_group = _source_group(located_fields)
    qc_issues = sorted({status for status in span_qc if status != "OK"})
    needs_review = auto_group == UNRESOLVED or bool(qc_issues)

    return {
        "Document_ID": normalize_document_id(document_id),
        "GT_Term": str(term_object.get("term") or ""),
        "Original_Entities": " | ".join(entity_texts),
        "Original_Labels": " | ".join(entity_labels),
        "Detected_Fields": " | ".join(
            field.name if field is not None else "UNMAPPED" for field in located_fields
        ),
        "Auto_Source_Group": auto_group,
        "Final_Source_Group": auto_group,
        "Review_Status": "HUMAN_REVIEW_REQUIRED" if needs_review else "AUTO_ACCEPT",
        "Span_QC": " | ".join(span_qc) if span_qc else "NO_ORIGINAL_ENTITY",
        "Review_Note": "",
    }


def classify_document_terms(
    gt_record: dict[str, Any],
    note_path: str | Path,
) -> list[dict[str, Any]]:
    """Classify every Phase 1 refined reference term for one document."""
    raw_content = Path(note_path).read_text(encoding="utf-8")
    source_text, field_spans = reconstruct_annotation_source(raw_content)
    return [
        classify_refined_term(
            document_id=str(gt_record.get("document_id") or Path(note_path).name),
            term_object=term_object,
            source_text=source_text,
            field_spans=field_spans,
        )
        for term_object in (gt_record.get("tab2_refined_terms") or [])
    ]
