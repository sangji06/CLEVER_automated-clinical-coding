from __future__ import annotations

import sys
import unittest
from pathlib import Path

ANALYSIS_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from source_attribution import (
    DIAGNOSIS_FIELD_ONLY,
    FIELD_PLUS_NARRATIVE,
    NARRATIVE_ONLY,
    classify_refined_term,
    reconstruct_annotation_source,
)


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_note = str(
            [
                {
                    "현병력-Free Text#10": "right-sided pain and fever",
                    "임상진단명 자동완성검색기능#29": "Abdominal pain",
                    "진단명-Free Text#30": "r/o appendicitis",
                }
            ]
        )
        self.source_text, self.field_spans = reconstruct_annotation_source(self.raw_note)

    def entity(self, text: str, label: str = "Diagnosis") -> dict:
        start = self.source_text.index(text)
        return {
            "entity": text,
            "label": label,
            "start": start,
            "end": start + len(text),
        }

    def classify(self, term: str, entities: list[dict]) -> dict:
        return classify_refined_term(
            document_id="synthetic_001.txt",
            term_object={"term": term, "original_entities": entities},
            source_text=self.source_text,
            field_spans=self.field_spans,
        )

    def test_diagnosis_fields_are_combined_into_one_source_group(self):
        result = self.classify(
            "appendicitis",
            [self.entity("Abdominal pain"), self.entity("r/o appendicitis")],
        )
        self.assertEqual(result["Auto_Source_Group"], DIAGNOSIS_FIELD_ONLY)
        self.assertEqual(result["Review_Status"], "AUTO_ACCEPT")

    def test_narrative_only_term(self):
        result = self.classify("fever", [self.entity("fever", "Symptom")])
        self.assertEqual(result["Auto_Source_Group"], NARRATIVE_ONLY)

    def test_field_plus_narrative_term(self):
        result = self.classify(
            "right-sided abdominal pain",
            [self.entity("Abdominal pain"), self.entity("right-sided", "Supportive")],
        )
        self.assertEqual(result["Auto_Source_Group"], FIELD_PLUS_NARRATIVE)

    def test_span_text_mismatch_is_flagged_for_review(self):
        entity = self.entity("fever", "Symptom")
        entity["entity"] = "cough"
        result = self.classify("cough", [entity])
        self.assertEqual(result["Auto_Source_Group"], NARRATIVE_ONLY)
        self.assertEqual(result["Review_Status"], "HUMAN_REVIEW_REQUIRED")
        self.assertIn("TEXT_MISMATCH", result["Span_QC"])

    def test_multiline_outer_quoted_export_uses_raw_offset_fallback(self):
        raw_note = (
            '"[{\'현병력-Free Text#10\': \'right-sided pain\nwith fever\', '
            "'임상진단명 자동완성검색기능#29': 'Abdominal pain', "
            "'진단명-Free Text#30': 'r/o appendicitis'}]\""
        )
        source_text, field_spans = reconstruct_annotation_source(raw_note)
        start = source_text.index("r/o appendicitis")
        result = classify_refined_term(
            document_id="synthetic_multiline.txt",
            term_object={
                "term": "appendicitis",
                "original_entities": [
                    {
                        "entity": "r/o appendicitis",
                        "label": "Diagnosis",
                        "start": start,
                        "end": start + len("r/o appendicitis"),
                    }
                ],
            },
            source_text=source_text,
            field_spans=field_spans,
        )
        self.assertEqual(result["Auto_Source_Group"], DIAGNOSIS_FIELD_ONLY)
        self.assertEqual(result["Review_Status"], "AUTO_ACCEPT")


if __name__ == "__main__":
    unittest.main()
