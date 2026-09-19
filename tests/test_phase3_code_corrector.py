import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from clever.phase3_code_corrector.rule_apply import KCDRuleProcessor


RULES_CSV = Path(__file__).resolve().parents[1] / "rules" / "code_correction_rules.csv"


def item(code, description):
    return {"term": description, "kcd_code": code, "kcd_description": description}


RULE_CASES = {
    1: ([item("R11.1", "Nausea"), item("R11.2", "Vomiting")], ["R11.3"]),
    2: ([item("R50.99", "Fever"), item("R68.8", "Chills")], ["R50.8"]),
    3: ([item("D64.9", "Anaemia"), item("C34.9", "Lung cancer")], ["C34.9", "D63.0"]),
    4: ([item("G40.90", "Epilepsy"), item("R56.9", "Convulsion")], ["G40.90"]),
    5: ([item("K80.20", "Gallbladder calculus"), item("K81.1", "Chronic cholecystitis")], ["K80.10"]),
    6: ([item("K80.20", "Gallbladder calculus"), item("K81.0", "Acute cholecystitis")], ["K80.00"]),
    7: ([item("K80.21", "Gallbladder calculus with obstruction"), item("K81.8", "Other cholecystitis")], ["K80.11"]),
    8: ([item("K80.21", "Gallbladder calculus with obstruction"), item("K81.0", "Acute cholecystitis")], ["K80.01"]),
    9: ([item("K80.50", "Bile duct calculus"), item("K83.0", "Cholangitis")], ["K80.30"]),
    10: ([item("K80.51", "Bile duct calculus with obstruction"), item("K83.0", "Cholangitis")], ["K80.31"]),
    11: ([item("K80.50", "Bile duct calculus"), item("K81.9", "Cholecystitis")], ["K80.40"]),
    12: ([item("K80.51", "Bile duct calculus with obstruction"), item("K81.0", "Acute cholecystitis")], ["K80.41"]),
    13: ([item("R11.3", "Nausea with vomiting"), item("R11.1", "Nausea")], ["R11.3"]),
    14: ([item("R11.3", "Nausea with vomiting"), item("R11.2", "Vomiting")], ["R11.3"]),
    15: ([item("R11.3", "Nausea with vomiting"), item("R11.1", "Nausea"), item("R11.2", "Vomiting")], ["R11.3"]),
    16: ([item("A09.9", "enteritis"), item("A09.9", "colitis")], ["A09.9"]),
    17: ([item("K58.8", "Irritable bowel syndrome"), item("A09.9", "diarrhea")], ["K58.1"]),
    18: ([item("A49.8", "Clostridium difficile infection"), item("A09.0", "Infectious gastroenteritis")], ["A04.7"]),
    19: ([item("I50.9", "Heart failure"), item("J81", "Pulmonary oedema")], ["I50.1"]),
    20: ([item("G03.9", "Meningitis"), item("B34.1", "enterovirus")], ["A87.0", "G02.0"]),
}


class Phase3CodeCorrectorTests(unittest.TestCase):
    def setUp(self):
        self.processor = KCDRuleProcessor()
        self.assertTrue(self.processor.load_rules(RULES_CSV))
        self.assertEqual(len(self.processor.rules), 20)

    def test_each_rule_positive_and_negative(self):
        for rule_id, (inputs, expected_codes) in RULE_CASES.items():
            with self.subTest(rule_id=rule_id):
                selected = next(
                    rule for rule in self.processor.rules if rule["rule_id"] == rule_id
                )
                all_rules = self.processor.rules
                self.processor.rules = [selected]

                output, logs = self.processor.apply_rules_to_record(
                    f"rule_{rule_id}_positive", inputs
                )
                self.assertEqual([log["rule_id"] for log in logs], [rule_id])
                self.assertEqual(
                    Counter(row["kcd_code"] for row in output),
                    Counter(expected_codes),
                )

                negative = inputs[:-1]
                negative_output, negative_logs = self.processor.apply_rules_to_record(
                    f"rule_{rule_id}_negative", negative
                )
                self.assertEqual(negative_logs, [])
                self.assertEqual(negative_output, negative)
                self.processor.rules = all_rules

    def test_specific_nausea_rule_has_priority(self):
        output, logs = self.processor.apply_rules_to_record(
            "triple_nausea",
            [
                item("R11.3", "Nausea with vomiting"),
                item("R11.1", "Nausea"),
                item("R11.2", "Vomiting"),
            ],
        )
        self.assertEqual([row["kcd_code"] for row in output], ["R11.3"])
        self.assertEqual([log["rule_id"] for log in logs], [15])

    def test_rule3_excludes_non_neoplasm_d50_d89(self):
        inputs = [item("D64.9", "Anaemia"), item("D50.9", "Iron deficiency anaemia")]
        output, logs = self.processor.apply_rules_to_record("blood_code_exclusion", inputs)
        self.assertEqual(output, inputs)
        self.assertEqual(logs, [])

    def test_rule16_removes_only_matched_rows(self):
        output, logs = self.processor.apply_rules_to_record(
            "three_a099_rows",
            [
                item("A09.9", "enteritis"),
                item("A09.9", "colitis"),
                item("A09.9", "diarrhea"),
            ],
        )
        self.assertEqual(
            Counter(row["kcd_code"] for row in output),
            Counter(["A09.9", "A09.9"]),
        )
        self.assertEqual([log["rule_id"] for log in logs], [16])

    def test_rule5_removes_pattern_matched_component(self):
        output, logs = self.processor.apply_rules_to_record(
            "gallstone_cholecystitis",
            [
                item("K80.20", "Gallbladder calculus"),
                item("K81.1", "Chronic cholecystitis"),
            ],
        )
        self.assertEqual([row["kcd_code"] for row in output], ["K80.10"])
        self.assertEqual([log["rule_id"] for log in logs], [5])

    def test_rule20_replaces_duplicates_and_assigns_both_labels(self):
        output, logs = self.processor.apply_rules_to_record(
            "enteroviral_meningitis",
            [
                item("G03.9", "Meningitis"),
                item("B34.1", "enterovirus"),
                item("A87.0", "old label"),
                item("G02.0", "old label"),
            ],
        )
        self.assertEqual([row["kcd_code"] for row in output], ["A87.0", "G02.0"])
        self.assertTrue(all(row["term"] == "Enteroviral meningitis" for row in output))
        self.assertEqual([log["rule_id"] for log in logs], [20])

    def test_process_json_writes_output_and_audit_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "phase2.json"
            output_path = temp_path / "phase3.json"
            log_path = temp_path / "phase3_rules.csv"
            input_path.write_text(
                json.dumps({"doc": RULE_CASES[2][0]}, ensure_ascii=False),
                encoding="utf-8",
            )

            self.processor.process_json(input_path, output_path, log_path)

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["kcd_code"] for row in result["doc"]], ["R50.8"])
            with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["rule_id"], "2")


if __name__ == "__main__":
    unittest.main()
