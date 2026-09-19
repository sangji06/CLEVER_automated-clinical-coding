from __future__ import annotations

import json
import unittest

from clever.direct_baseline.prompts import PROMPT_VERSION, build_direct_kcd_prompt
from clever.direct_baseline.runner import (
    _create_context_text,
    normalize_predictions,
    parse_direct_output,
    parse_direct_output_with_metadata,
    process_record_direct,
    run_direct_baseline_dir,
)


class FakeBedrockClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [{"text": next(self.responses)}],
                }
            }
        }


class DirectBaselineTests(unittest.TestCase):
    def test_multiline_outer_quoted_note_is_formatted_by_field(self):
        record = '"[{\'주증상-Text#7\': \'Fever\nfor two days\', \'진단명-Free Text#30\': \'URI\'}]"'

        formatted = _create_context_text(record)

        self.assertIn("### 주증상-Text#7\nFever\nfor two days", formatted)
        self.assertIn("### 진단명-Free Text#30\nURI", formatted)
        self.assertFalse(formatted.startswith('"['))

    def test_prompt_inserts_note_without_using_pipeline_outputs(self):
        prompt = build_direct_kcd_prompt("chief complaint: fever")

        self.assertIn("chief complaint: fever", prompt)
        self.assertNotIn("Phase 1", prompt)
        self.assertNotIn("PREVIOUS_STEP_RESULTS", prompt)
        self.assertNotIn("vector", prompt.lower())
        self.assertIn("generic WHO ICD-10", prompt)
        self.assertIn("R50.99", prompt)
        self.assertIn("Every eligible diagnosis and symptom", prompt)
        self.assertIn("C61, not C61.90", prompt)
        self.assertIn("Do not assume that every code requires a decimal point", prompt)

    def test_parse_direct_output_extracts_balanced_fenced_json(self):
        response = (
            "```json\n"
            '[{"diagnosis":"Fever","kcd_code":"R50.9"}]\n'
            "```\nThis trailing text is ignored."
        )

        parsed = parse_direct_output(response)

        self.assertEqual(parsed, [{"diagnosis": "Fever", "kcd_code": "R50.9"}])

    def test_parse_direct_output_distinguishes_valid_empty_list_from_failure(self):
        self.assertEqual(parse_direct_output("[]"), [])
        with self.assertRaises(ValueError):
            parse_direct_output("ERROR")

    def test_parse_direct_output_repairs_only_missing_top_level_object_opener(self):
        malformed = """[
          {"diagnosis": "Fever", "kcd_code": "R50.9"},
          "diagnosis": "Cough", "kcd_code": "R05"},
          {"diagnosis": "Vomiting", "kcd_code": "R11"}
        ]"""

        parsed, repairs = parse_direct_output_with_metadata(malformed)

        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[1], {"diagnosis": "Cough", "kcd_code": "R05"})
        self.assertEqual(len(repairs), 1)

    def test_normalize_predictions_aligns_with_phase3_evaluation_schema(self):
        normalized, warnings = normalize_predictions([
            {"diagnosis": "Fever", "kcd_code": " r50.9 "},
            {"diagnosis": "Fever", "kcd_code": "R50.9"},
            {"diagnosis": "Depression", "kcd_code": "F32.9"},
            {"diagnosis": "Bad code", "kcd_code": "not-a-code"},
        ])

        self.assertEqual(normalized, [{
            "term": "Fever",
            "kcd_code": "R50.9",
            "kcd_description": "",
        }])
        self.assertEqual(len(warnings), 3)

    def test_process_record_direct_uses_frozen_inference_settings(self):
        client = FakeBedrockClient([
            '[{"diagnosis":"Acute gastroenteritis","kcd_code":"A09.9"}]'
        ])

        predictions, audit = process_record_direct(
            client=client,
            model_id="qwen.qwen3-32b-v1:0",
            record="[{'진단명-Free Text#30': 'AGE'}]",
            filename="synthetic_001.txt",
        )

        self.assertEqual(predictions[0]["kcd_code"], "A09.9")
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(audit["prompt_version"], PROMPT_VERSION)
        request = client.calls[0]
        self.assertEqual(request["inferenceConfig"], {
            "maxTokens": 4096,
            "temperature": 0.0,
            "topP": 0.9,
        })
        self.assertIn("AGE", request["messages"][0]["content"][0]["text"])

    def test_directory_runner_writes_evaluation_output_audit_and_manifest(self):
        with self.subTest("synthetic directory run"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_dir = temp_path / "synthetic_notes"
                input_dir.mkdir()
                (input_dir / "record_001.txt").write_text(
                    "[{'진단명-Free Text#30': 'Fever'}]", encoding="utf-8"
                )
                output_path = temp_path / "direct_predictions.json"
                client = FakeBedrockClient([
                    '[{"diagnosis":"Fever","kcd_code":"R50.9"}]'
                ])

                predictions = run_direct_baseline_dir(
                    input_dir=input_dir,
                    output_json=output_path,
                    model_id="qwen.qwen3-32b-v1:0",
                    batch_size=1,
                    client=client,
                )

                saved_predictions = json.loads(output_path.read_text(encoding="utf-8"))
                audit = json.loads(
                    (temp_path / "direct_predictions_audit.json").read_text(encoding="utf-8")
                )
                manifest = json.loads(
                    (temp_path / "direct_predictions_manifest.json").read_text(encoding="utf-8")
                )

                self.assertEqual(predictions, saved_predictions)
                self.assertEqual(saved_predictions["record_001"][0]["kcd_code"], "R50.9")
                self.assertEqual(audit["record_001"]["status"], "completed")
                self.assertEqual(manifest["selected_record_count"], 1)
                self.assertTrue(manifest["single_run_per_record"])
                self.assertEqual(manifest["prompt_version"], PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
