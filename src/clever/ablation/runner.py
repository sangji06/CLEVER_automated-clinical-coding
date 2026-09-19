"""Shared runners for CLEVER control experiments."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from clever.ablation.no_guideline import convert_no_guideline_to_phase2_input
from clever.phase2_code_retriever.pipeline import run_phase2_retrieval_only
from clever.phase3_code_corrector.rule_apply import KCDRuleProcessor


def run_phase2_and_phase3(
    phase1_output: str | Path,
    faiss_index: str | Path,
    rules_csv: str | Path,
    phase2_output: str | Path,
    phase3_output: str | Path,
    rule_log: str | Path,
    phase1_schema: str = "guided",
) -> None:
    """Run shared Phase 2 retrieval and Phase 3 rule correction.

    `phase1_schema` may be:
    - `guided`: standard CLEVER Phase 1 output with `step4_expansion`.
    - `no_guideline`: control Phase 1 output with `baseline_extraction`.
    """
    phase1_output = Path(phase1_output)
    phase2_output = Path(phase2_output)
    phase3_output = Path(phase3_output)
    rule_log = Path(rule_log)

    phase2_output.parent.mkdir(parents=True, exist_ok=True)
    phase3_output.parent.mkdir(parents=True, exist_ok=True)
    rule_log.parent.mkdir(parents=True, exist_ok=True)

    if phase1_schema == "guided":
        phase2_input = phase1_output
        temp_dir = None
    elif phase1_schema == "no_guideline":
        temp_dir = tempfile.TemporaryDirectory()
        phase2_input = Path(temp_dir.name) / "phase1_no_guideline_for_phase2.json"
        with open(phase1_output, "r", encoding="utf-8") as file:
            no_guideline_data = json.load(file)
        converted = convert_no_guideline_to_phase2_input(no_guideline_data)
        phase2_input.write_text(json.dumps(converted, ensure_ascii=False, indent=4), encoding="utf-8")
    else:
        raise ValueError("phase1_schema must be either 'guided' or 'no_guideline'.")

    try:
        run_phase2_retrieval_only(
            faiss_index_path=str(faiss_index),
            model_output_path=str(phase2_input),
            json_output_path=str(phase2_output),
        )

        processor = KCDRuleProcessor()
        if not processor.load_rules(rules_csv):
            raise RuntimeError(f"Failed to load rule CSV: {rules_csv}")

        processor.process_json(
            input_json_path=str(phase2_output),
            output_json_path=str(phase3_output),
            log_csv_path=str(rule_log),
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
