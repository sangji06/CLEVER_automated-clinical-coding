"""No-guideline Phase 1 extractor used for CLEVER control experiments.

This module preserves the core logic of the original
`Ablation Study/baseline1_no_guideline.py` script:

- one zero-shot prompt,
- no guideline-informed multi-step extraction,
- Bedrock Converse API call,
- JSON parsing,
- source-text span realignment,
- output under the original `baseline_extraction` key.

Public-release changes are limited to packaging the code as reusable functions
and replacing hard-coded paths/model settings with function or CLI arguments.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from clever.phase1_guided_extractor.extractor import (
    create_context_text,
    initialize_bedrock_client,
    load_medical_records_from_folder,
    parse_bedrock_output,
    realign_simple_indices,
    to_source_text_like_streamlit,
)


NO_GUIDELINE_PROMPT = """You are a clinical coding assistant.
    Read the following emergency department note and extract the 'Diagnosis' and 'Symptom' terms required for KCD clinical coding.
    **If a term is written as an abbreviation, convert it to its full term.**
    ex) 'URI' → 'Upper respiratory infection', 'AGE' → 'Acute gastroenteritis'
    **If a term is written Korean, translate it into English.**
    ex) '발열' → 'Fever', '구토' → 'Vomiting'
    You must output the result ONLY in the JSON format below.
    [
        {
            "entity": "extracted_term",
            "label": "Diagnosis" or "Symptom"
        }
    ]
    """


def call_bedrock_no_guideline(
    client,
    model_id: str,
    prompt: str,
    context_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """Call Bedrock with the original no-guideline baseline prompt pattern."""
    user_prompt = (
        f"{prompt}\n\n"
        f"---\n"
        f"Here is the full medical record for context:\n\n"
        f"### FULL_CONTEXT_RECORD ###\n{context_text}\n\n"
        f"---\n"
        f"CRITICAL: Output ONLY JSON.\n"
        f"Output format: [{{'entity': 'extracted term', 'label': 'Diagnosis or Symptom'}}]\n"
        f"Output:"
    )

    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    inference_config = {"maxTokens": max_tokens, "temperature": temperature, "topP": top_p}

    for attempt in range(3):
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
            )
            return response["output"]["message"]["content"][0]["text"]
        except (ClientError, Exception) as error:
            print(f"[Attempt {attempt + 1}] Error: {error}")
            if attempt < 2:
                time.sleep(2)

    return "ERROR"


def process_record_no_guideline(
    client,
    model_id: str,
    record: str,
    filename: str = "",
    prompt: str = NO_GUIDELINE_PROMPT,
) -> dict[str, Any]:
    """Run the original no-guideline single-step extraction for one record."""
    results: dict[str, Any] = {
        "filename": filename,
        "status": "processing",
        "baseline_extraction": None,
        "error": None,
    }

    try:
        source_text = to_source_text_like_streamlit(record)
        context_text = create_context_text(record)

        response_text = call_bedrock_no_guideline(
            client=client,
            model_id=model_id,
            prompt=prompt,
            context_text=context_text,
        )
        parsed_result = parse_bedrock_output(response_text)
        results["baseline_extraction"] = realign_simple_indices(parsed_result, source_text)
        results["status"] = "completed"
        return results
    except Exception as error:
        results["status"] = "failed"
        results["error"] = str(error)
        return results


def process_records_no_guideline(
    client,
    model_id: str,
    filenames: list[str],
    records: list[str],
    max_workers: int = 1,
) -> dict[str, dict[str, Any]]:
    """Process multiple records with the no-guideline baseline extractor."""
    output: dict[str, dict[str, Any]] = {}

    if max_workers <= 1:
        for filename, record in zip(filenames, records):
            output[Path(filename).stem] = process_record_no_guideline(
                client=client,
                model_id=model_id,
                record=record,
                filename=filename,
            )
        return output

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_record_no_guideline,
                client,
                model_id,
                record,
                filename,
            ): filename
            for filename, record in zip(filenames, records)
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                output[Path(filename).stem] = future.result(timeout=600)
            except Exception as error:
                output[Path(filename).stem] = {"status": "failed", "error": str(error)}

    return output


def save_batch_results(
    batch_results: dict[str, dict[str, Any]],
    batch_num: int,
    output_dir: str | Path,
) -> Path:
    """Save one no-guideline batch result using the original filename pattern."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_file = output_dir / f"baseline_batch_{batch_num:03d}.json"
    batch_file.write_text(json.dumps(batch_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return batch_file


def load_progress(output_dir: str | Path) -> set[str]:
    """Load processed record ids from existing no-guideline batch files."""
    output_dir = Path(output_dir)
    processed_files: set[str] = set()

    if not output_dir.exists():
        return processed_files

    for batch_file in sorted(output_dir.glob("baseline_batch_*.json")):
        with open(batch_file, "r", encoding="utf-8") as file:
            batch_data = json.load(file)
        processed_files.update(batch_data.keys())

    return processed_files


def merge_all_results(output_dir: str | Path, final_output_path: str | Path) -> dict[str, dict[str, Any]]:
    """Merge no-guideline batch files into the final output JSON."""
    output_dir = Path(output_dir)
    all_results: dict[str, dict[str, Any]] = {}

    for batch_file in sorted(output_dir.glob("baseline_batch_*.json")):
        with open(batch_file, "r", encoding="utf-8") as file:
            batch_data = json.load(file)
        all_results.update(batch_data)

    final_output_path = Path(final_output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=4), encoding="utf-8")
    return all_results


def run_no_guideline_extraction_dir(
    input_dir: str | Path,
    output_json: str | Path,
    model_id: str,
    region_name: str = "us-west-2",
    max_workers: int = 1,
    batch_size: int = 50,
    batch_dir: str | Path | None = None,
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run control Phase 1 on all `.txt` records in a directory.

    Batch files and resume behavior follow the original baseline script. If
    `batch_dir` is omitted, batches are saved next to `output_json`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    client = initialize_bedrock_client(region_name=region_name)
    filenames, records = load_medical_records_from_folder(input_dir)

    output_json = Path(output_json)
    batch_dir = Path(batch_dir) if batch_dir is not None else output_json.parent / "baseline_results"
    batch_dir.mkdir(parents=True, exist_ok=True)

    existing_batches = list(batch_dir.glob("baseline_batch_*.json"))
    if existing_batches and not resume:
        raise ValueError(
            "Existing baseline_batch_*.json files found. Use --resume or choose a new --batch-dir."
        )

    processed_files = load_progress(batch_dir) if resume else set()
    remaining_pairs = [
        (filename, record)
        for filename, record in zip(filenames, records)
        if Path(filename).stem not in processed_files
    ]

    next_batch_num = len(existing_batches) + 1 if resume else 1

    for start in range(0, len(remaining_pairs), batch_size):
        batch_pairs = remaining_pairs[start : start + batch_size]
        batch_filenames = [item[0] for item in batch_pairs]
        batch_records = [item[1] for item in batch_pairs]

        batch_results = process_records_no_guideline(
            client=client,
            model_id=model_id,
            filenames=batch_filenames,
            records=batch_records,
            max_workers=max_workers,
        )
        save_batch_results(batch_results, next_batch_num, batch_dir)
        next_batch_num += 1

        if start + batch_size < len(remaining_pairs):
            time.sleep(2)

    return merge_all_results(batch_dir, output_json)


def convert_no_guideline_to_phase2_input(
    no_guideline_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Adapt `baseline_extraction` output to the existing Phase 2 input schema.

    This does not change extracted terms. It only wraps each baseline entity as
    a `step4_expansion` term so the shared Phase 2 retriever can be reused.
    """
    converted: dict[str, dict[str, Any]] = {}

    for record_id, record in no_guideline_data.items():
        baseline_items = record.get("baseline_extraction") or []
        step4_items = []
        for item in baseline_items:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            if not entity:
                continue
            step4_items.append({
                "term": entity,
                "original_entities": [item],
            })

        converted_record = dict(record)
        converted_record["step4_expansion"] = step4_items
        converted[record_id] = converted_record

    return converted


def write_phase2_input_from_no_guideline(
    no_guideline_json: str | Path,
    output_json: str | Path,
) -> None:
    """Write an adapted Phase 2 input JSON from no-guideline Phase 1 output."""
    with open(no_guideline_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    converted = convert_no_guideline_to_phase2_input(data)

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(converted, ensure_ascii=False, indent=4), encoding="utf-8")
