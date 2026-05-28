import csv
import json

from clever.phase3_code_corrector.rule_apply import KCDRuleProcessor


def test_original_phase3_rule_processor_combines_fever_and_chills(tmp_path):
    rules_csv = tmp_path / "rules.csv"
    input_json = tmp_path / "phase2.json"
    output_json = tmp_path / "phase3.json"
    log_csv = tmp_path / "rules_log.csv"

    with open(rules_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rule_id",
                "exact_codes",
                "exact_descriptions",
                "pattern_codes",
                "pattern_descriptions",
                "action_type",
                "remove_codes",
                "add_code",
                "add_diagnosis",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "rule_id": "synthetic",
            "exact_codes": "R50.99,R68.8",
            "exact_descriptions": ",*chill*",
            "pattern_codes": "",
            "pattern_descriptions": "",
            "action_type": "replace",
            "remove_codes": "R50.99,R68.8",
            "add_code": "R50.8",
            "add_diagnosis": "Fever with chills",
            "description": "Synthetic fever and chills rule",
        })

    with open(input_json, "w", encoding="utf-8") as file:
        json.dump({
            "synthetic_record": [
                {"term": "Fever", "kcd_code": "R50.99", "kcd_description": "Fever"},
                {"term": "Chills", "kcd_code": "R68.8", "kcd_description": "Chill"},
            ]
        }, file)

    processor = KCDRuleProcessor()
    assert processor.load_rules(rules_csv)
    processor.process_json(input_json, output_json, log_csv)

    with open(output_json, "r", encoding="utf-8") as file:
        corrected = json.load(file)

    assert corrected["synthetic_record"] == [
        {
            "term": "Fever with chills",
            "kcd_code": "R50.8",
            "kcd_description": "Fever with chills",
        }
    ]
