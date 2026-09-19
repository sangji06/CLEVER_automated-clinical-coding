from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ANALYSIS_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from analyze_phase1_module11_source_recall import (
    NARRATIVE_ONLY,
    classify_tab1_terms,
    load_manual_source_overrides,
    metrics,
    normalize_term,
    percentile,
    prediction_terms,
)


class SourceRecallTests(unittest.TestCase):
    def test_normalization_matches_module11_endpoint(self):
        self.assertEqual(normalize_term(" Acute\u3000Appendicitis "), "acuteappendicitis")

    def test_step3_prediction_terms_are_unique(self):
        prediction = {
            "step3_filtering": [
                {"term": "Fever"},
                {"entity": " fever "},
                {"term": "Cough"},
            ]
        }
        self.assertEqual(prediction_terms(prediction), {"fever", "cough"})

    def test_linear_percentile(self):
        values = [0.0, 0.5, 1.0]
        self.assertAlmostEqual(percentile(values, 0.25), 0.25)
        self.assertAlmostEqual(percentile(values, 0.975), 0.975)

    def test_metrics(self):
        precision, recall, f1 = metrics(tp=3, fp=1, fn=3)
        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.6)

    def test_private_override_csv_resolves_unmapped_term(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overrides.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["Document_ID", "GT_Term", "Source_Group", "Review_Note"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Document_ID": "1_Short_1",
                        "GT_Term": "Headache",
                        "Source_Group": NARRATIVE_ONLY,
                        "Review_Note": "Confirmed locally",
                    }
                )
            overrides = load_manual_source_overrides(path)
            gt = {
                "1_Short_1": {
                    "tab1_annotations": [
                        {"entity": "Headache", "start": 10, "end": 18}
                    ]
                }
            }
            rows, groups = classify_tab1_terms(gt, {}, overrides)
            self.assertEqual(groups["1_Short_1"]["headache"], NARRATIVE_ONLY)
            self.assertEqual(rows[0]["Review_Status"], "USER_CONFIRMED_SOURCE")


if __name__ == "__main__":
    unittest.main()
