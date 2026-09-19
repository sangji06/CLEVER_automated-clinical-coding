"""Run a direct raw-note-to-final-KCD LLM baseline.

This baseline is intentionally separate from CLEVER's three phases. It sends
each raw note to one LLM prompt and saves the returned final KCD codes in the
same document-to-code-list schema consumed by the cumulative Phase 3 evaluator.
It does not use Phase 1 output, vector retrieval, a curated KCD database, or
Phase 3 post-processing rules.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clever.direct_baseline.prompts import (
    DIRECT_KCD_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_direct_kcd_prompt,
    prompt_sha256,
)


DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_ATTEMPTS = 3
INPUT_FORMATTER_VERSION = "note-fields-v1.1"
EXCLUDED_CODE_PREFIXES = {"F", "V", "W", "X", "Y"}
KCD_CODE_PATTERN = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")
SERIALIZED_FIELD_PATTERN = re.compile(r"'([^'\n]+#\d+)'\s*:\s*'")


def _extract_tolerant_note_fields(raw_content: str) -> dict[str, str]:
    """Extract named note fields from quoted, multiline pseudo-literals.

    Some private exports wrap the entire Python-like list in double quotes and
    contain literal newlines inside single-quoted field values. Such files are
    readable but are not valid inputs to ``ast.literal_eval``. Field boundaries
    remain explicit because every key ends in ``#<number>``; this fallback uses
    only those boundaries and preserves each field value verbatim.
    """
    text = raw_content.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]

    matches = list(SERIALIZED_FIELD_PATTERN.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end]
        if index + 1 < len(matches):
            value = re.sub(r"'\s*,\s*$", "", value)
        else:
            value = re.sub(r"'\s*}\s*]\s*$", "", value)
        fields[match.group(1)] = value
    return fields


def _create_context_text(raw_content: str) -> str:
    """Format a serialized note dictionary without using any pipeline output."""
    try:
        data = ast.literal_eval(raw_content)
        if isinstance(data, str) and data != raw_content:
            return _create_context_text(data)
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            data = None
    except (SyntaxError, ValueError):
        data = None

    fields = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
    if fields is None:
        fields = _extract_tolerant_note_fields(raw_content)
    if not fields:
        return raw_content
    return "\n\n".join(
        f"### {key}\n{value}"
        for key, value in fields.items()
        if value is not None and str(value).strip()
    )


def _load_medical_records(input_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load sorted `.txt` notes from one explicitly supplied directory."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths = sorted(input_dir.glob("*.txt"))
    return [path.name for path in paths], [path.read_text(encoding="utf-8") for path in paths]


def _extract_first_balanced_json(text: str) -> str:
    """Extract the first balanced JSON array or object from model text."""
    start = next((index for index, char in enumerate(text) if char in "[{"), -1)
    if start < 0:
        raise ValueError("The response did not contain a JSON array or object.")

    stack: list[str] = []
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                raise ValueError("The response contained unbalanced JSON delimiters.")
            expected = "[" if char == "]" else "{"
            if stack[-1] != expected:
                raise ValueError("The response contained mismatched JSON delimiters.")
            stack.pop()
            if not stack:
                return text[start : index + 1]

    raise ValueError("The response contained incomplete JSON.")


def _repair_missing_top_level_object_openers(text: str) -> tuple[str, list[str]]:
    """Repair a missing ``{`` before a key directly inside the output list.

    The expected schema is a list of objects. This repair is deliberately
    narrow: it only inserts an object opener when a quoted token appears at the
    top level of that list and is immediately followed by a colon. It does not
    infer, delete, or alter any diagnosis or code value.
    """
    start = next((index for index, char in enumerate(text) if char in "[{"), -1)
    if start < 0 or text[start] != "[":
        return text, []

    repaired: list[str] = list(text[:start])
    stack: list[str] = []
    repairs: list[str] = []
    in_string = False
    escaped = False
    index = start

    while index < len(text):
        char = text[index]
        if in_string:
            repaired.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            if stack == ["["]:
                closing = index + 1
                string_escaped = False
                while closing < len(text):
                    candidate = text[closing]
                    if string_escaped:
                        string_escaped = False
                    elif candidate == "\\":
                        string_escaped = True
                    elif candidate == '"':
                        break
                    closing += 1
                after = closing + 1
                while after < len(text) and text[after].isspace():
                    after += 1
                if closing < len(text) and after < len(text) and text[after] == ":":
                    repaired.append("{")
                    stack.append("{")
                    repairs.append(
                        f"inserted missing object opener before top-level key at character {index}"
                    )
            repaired.append(char)
            in_string = True
        elif char in "[{":
            repaired.append(char)
            stack.append(char)
        elif char in "]}":
            repaired.append(char)
            if stack:
                expected = "[" if char == "]" else "{"
                if stack[-1] == expected:
                    stack.pop()
        else:
            repaired.append(char)
        index += 1

    return "".join(repaired), repairs


def parse_direct_output_with_metadata(response_text: str) -> tuple[list[Any], list[str]]:
    """Parse a response and report any deterministic syntax repairs applied."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("The model response was empty.")
    if response_text.strip() == "ERROR":
        raise ValueError("The Bedrock request failed after all retry attempts.")

    stripped = response_text.strip()
    repairs: list[str] = []
    try:
        payload_text = _extract_first_balanced_json(stripped)
    except ValueError as original_error:
        repaired_text, repairs = _repair_missing_top_level_object_openers(stripped)
        if not repairs:
            raise original_error
        payload_text = _extract_first_balanced_json(repaired_text)

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError) as error:
            raise ValueError("The model response was not valid JSON-like data.") from error

    if isinstance(payload, dict):
        for key in ("predictions", "results", "codes"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if any(key in payload for key in ("diagnosis", "term", "kcd_code", "code")):
                payload = [payload]

    if not isinstance(payload, list):
        raise ValueError("The parsed response was not a list of predictions.")
    return payload, repairs


def parse_direct_output(response_text: str) -> list[Any]:
    """Parse a direct-baseline response while distinguishing failure from []."""
    payload, _ = parse_direct_output_with_metadata(response_text)
    return payload


def normalize_kcd_code(value: Any) -> str:
    """Normalize and validate one model-generated KCD code."""
    if value is None:
        return ""
    code = re.sub(r"\s+", "", str(value)).upper()
    if not KCD_CODE_PATTERN.fullmatch(code):
        return ""
    if code[0] in EXCLUDED_CODE_PREFIXES:
        return ""
    return code


def normalize_predictions(items: list[Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Convert model output to the shared evaluation schema and report drops."""
    predictions: list[dict[str, str]] = []
    warnings: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(f"item {index}: ignored because it was not an object")
            continue

        diagnosis = str(
            item.get("diagnosis")
            or item.get("term")
            or item.get("Diagnosis")
            or ""
        ).strip()
        raw_code = item.get("kcd_code") or item.get("code") or item.get("KCD Code")
        code = normalize_kcd_code(raw_code)

        if not diagnosis:
            warnings.append(f"item {index}: ignored because diagnosis was empty")
            continue
        if not code:
            warnings.append(f"item {index}: ignored because KCD code was invalid or out of scope")
            continue

        pair = (diagnosis.casefold(), code)
        if pair in seen_pairs:
            warnings.append(f"item {index}: ignored duplicate diagnosis-code pair")
            continue
        seen_pairs.add(pair)

        predictions.append({
            "term": diagnosis,
            "kcd_code": code,
            "kcd_description": "",
        })

    return predictions, warnings


def _response_text(response: dict[str, Any]) -> str:
    """Collect textual answer blocks from a Bedrock Converse response."""
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = [str(block.get("text", "")) for block in blocks if block.get("text")]
    if text_parts:
        return "".join(text_parts).strip()

    reasoning_parts: list[str] = []
    for block in blocks:
        reasoning = block.get("reasoningContent", {})
        reasoning_text = reasoning.get("reasoningText", {}) if isinstance(reasoning, dict) else {}
        if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
            reasoning_parts.append(str(reasoning_text["text"]))
    return "".join(reasoning_parts).strip()


def call_bedrock_direct(
    client,
    model_id: str,
    clinical_note: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> str:
    """Call Bedrock once per attempt with the frozen direct KCD prompt."""
    messages = [{
        "role": "user",
        "content": [{"text": build_direct_kcd_prompt(clinical_note)}],
    }]
    system = [{"text": DIRECT_KCD_SYSTEM_PROMPT}]
    inference_config = {
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topP": top_p,
    }

    for attempt in range(max_attempts):
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
            )
            output_text = _response_text(response)
            if output_text:
                return output_text
            raise ValueError("Bedrock returned no textual response block.")
        except Exception as error:
            print(f"[Direct baseline attempt {attempt + 1}] Error: {error}")
            if attempt + 1 < max_attempts:
                time.sleep(2)
    return "ERROR"


def process_record_direct(
    client,
    model_id: str,
    record: str,
    filename: str = "",
    save_raw_response: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Generate final KCD predictions and an audit entry for one raw note."""
    audit: dict[str, Any] = {
        "filename": filename,
        "status": "processing",
        "prompt_version": PROMPT_VERSION,
        "input_formatter_version": INPUT_FORMATTER_VERSION,
        "prediction_count": 0,
        "validation_warnings": [],
        "json_repair_applied": False,
        "json_repairs": [],
        "response_sha256": None,
        "error": None,
    }

    try:
        clinical_note = _create_context_text(record)
        response_text = call_bedrock_direct(
            client=client,
            model_id=model_id,
            clinical_note=clinical_note,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_attempts=max_attempts,
        )
        audit["response_sha256"] = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        if save_raw_response:
            audit["raw_response"] = response_text

        parsed, repairs = parse_direct_output_with_metadata(response_text)
        predictions, warnings = normalize_predictions(parsed)
        audit["status"] = "completed"
        audit["prediction_count"] = len(predictions)
        audit["validation_warnings"] = warnings
        audit["json_repair_applied"] = bool(repairs)
        audit["json_repairs"] = repairs
        return predictions, audit
    except Exception as error:
        audit["status"] = "failed"
        audit["error"] = str(error)
        return [], audit


def process_records_direct(
    client,
    model_id: str,
    filenames: list[str],
    records: list[str],
    max_workers: int = 1,
    save_raw_responses: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    """Process raw notes, preserving deterministic record ids and audit data."""
    predictions: dict[str, list[dict[str, str]]] = {}
    audit: dict[str, dict[str, Any]] = {}

    def run_one(filename: str, record: str):
        return process_record_direct(
            client=client,
            model_id=model_id,
            record=record,
            filename=filename,
            save_raw_response=save_raw_responses,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_attempts=max_attempts,
        )

    if max_workers <= 1:
        for filename, record in zip(filenames, records):
            record_id = Path(filename).stem
            predictions[record_id], audit[record_id] = run_one(filename, record)
        return predictions, audit

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, filename, record): filename
            for filename, record in zip(filenames, records)
        }
        for future in as_completed(futures):
            filename = futures[future]
            record_id = Path(filename).stem
            try:
                predictions[record_id], audit[record_id] = future.result(timeout=600)
            except Exception as error:
                predictions[record_id] = []
                audit[record_id] = {
                    "filename": filename,
                    "status": "failed",
                    "prompt_version": PROMPT_VERSION,
                    "input_formatter_version": INPUT_FORMATTER_VERSION,
                    "prediction_count": 0,
                    "validation_warnings": [],
                    "json_repair_applied": False,
                    "json_repairs": [],
                    "response_sha256": None,
                    "error": str(error),
                }
    return predictions, audit


def _batch_paths(batch_dir: Path, batch_num: int) -> tuple[Path, Path]:
    return (
        batch_dir / f"direct_kcd_batch_{batch_num:03d}.json",
        batch_dir / f"direct_kcd_audit_batch_{batch_num:03d}.json",
    )


def save_batch_results(
    predictions: dict[str, list[dict[str, str]]],
    audit: dict[str, dict[str, Any]],
    batch_num: int,
    batch_dir: str | Path,
) -> tuple[Path, Path]:
    """Save evaluation-compatible predictions and separate audit information."""
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, audit_path = _batch_paths(batch_dir, batch_num)
    prediction_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return prediction_path, audit_path


def load_completed_records(batch_dir: str | Path) -> set[str]:
    """Load successfully completed ids; technical failures remain retryable."""
    completed: set[str] = set()
    for audit_path in sorted(Path(batch_dir).glob("direct_kcd_audit_batch_*.json")):
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        completed.update(
            record_id
            for record_id, entry in data.items()
            if isinstance(entry, dict) and entry.get("status") == "completed"
        )
    return completed


def _merge_batches(batch_dir: Path, pattern: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for batch_path in sorted(batch_dir.glob(pattern)):
        merged.update(json.loads(batch_path.read_text(encoding="utf-8")))
    return merged


def _next_batch_number(batch_dir: Path) -> int:
    numbers = []
    for path in batch_dir.glob("direct_kcd_batch_*.json"):
        match = re.search(r"_(\d+)\.json$", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _run_config(
    model_id: str,
    region_name: str,
    input_dir: str | Path,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
    selected_record_ids: list[str],
) -> dict[str, Any]:
    record_id_text = "\n".join(selected_record_ids)
    return {
        "baseline": "direct_raw_note_to_final_kcd",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "input_formatter_version": INPUT_FORMATTER_VERSION,
        "model_id": model_id,
        "region": region_name,
        "input_dir": str(Path(input_dir).resolve()),
        "inference_config": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
        "technical_max_attempts": max_attempts,
        "selected_record_count": len(selected_record_ids),
        "selected_record_ids_sha256": hashlib.sha256(
            record_id_text.encode("utf-8")
        ).hexdigest(),
    }


def _prepare_run_config(batch_dir: Path, config: dict[str, Any], resume: bool) -> None:
    config_path = batch_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                "Existing direct-baseline batches were created with a different prompt, model, "
                "input directory, or inference setting. Choose a new --batch-dir."
            )
        if not resume and list(batch_dir.glob("direct_kcd_batch_*.json")):
            raise ValueError(
                "Existing direct_kcd_batch_*.json files found. Use --resume or choose a new --batch-dir."
            )
        return
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def run_direct_baseline_dir(
    input_dir: str | Path,
    output_json: str | Path,
    model_id: str,
    region_name: str = "us-west-2",
    max_workers: int = 1,
    batch_size: int = 50,
    batch_dir: str | Path | None = None,
    audit_json: str | Path | None = None,
    manifest_json: str | Path | None = None,
    resume: bool = True,
    save_raw_responses: bool = False,
    max_records: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    client=None,
) -> dict[str, list[dict[str, str]]]:
    """Run the frozen direct baseline on all sorted `.txt` notes in a folder."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be greater than 0 when provided.")

    output_json = Path(output_json)
    batch_dir = (
        Path(batch_dir)
        if batch_dir is not None
        else output_json.parent / f"{output_json.stem}_batches"
    )
    audit_json = (
        Path(audit_json)
        if audit_json is not None
        else output_json.with_name(f"{output_json.stem}_audit.json")
    )
    manifest_json = (
        Path(manifest_json)
        if manifest_json is not None
        else output_json.with_name(f"{output_json.stem}_manifest.json")
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    filenames, records = _load_medical_records(input_dir)
    if max_records is not None:
        filenames = filenames[:max_records]
        records = records[:max_records]

    run_config = _run_config(
        model_id=model_id,
        region_name=region_name,
        input_dir=input_dir,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_attempts=max_attempts,
        selected_record_ids=[Path(filename).stem for filename in filenames],
    )
    _prepare_run_config(batch_dir, run_config, resume=resume)

    completed = load_completed_records(batch_dir) if resume else set()
    remaining = [
        (filename, record)
        for filename, record in zip(filenames, records)
        if Path(filename).stem not in completed
    ]

    if remaining and client is None:
        from clever.phase1_guided_extractor.extractor import initialize_bedrock_client

        client = initialize_bedrock_client(region_name=region_name)

    batch_num = _next_batch_number(batch_dir)
    for start in range(0, len(remaining), batch_size):
        pairs = remaining[start : start + batch_size]
        batch_predictions, batch_audit = process_records_direct(
            client=client,
            model_id=model_id,
            filenames=[pair[0] for pair in pairs],
            records=[pair[1] for pair in pairs],
            max_workers=max_workers,
            save_raw_responses=save_raw_responses,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_attempts=max_attempts,
        )
        save_batch_results(batch_predictions, batch_audit, batch_num, batch_dir)
        batch_num += 1

    predictions = _merge_batches(batch_dir, "direct_kcd_batch_*.json")
    audit = _merge_batches(batch_dir, "direct_kcd_audit_batch_*.json")

    selected_ids = {Path(filename).stem for filename in filenames}
    predictions = {key: value for key, value in predictions.items() if key in selected_ids}
    audit = {key: value for key, value in audit.items() if key in selected_ids}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    completed_count = sum(entry.get("status") == "completed" for entry in audit.values())
    failed_count = sum(entry.get("status") == "failed" for entry in audit.values())
    manifest = {
        **run_config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_record_count": len(filenames),
        "completed_record_count": completed_count,
        "failed_record_count": failed_count,
        "single_run_per_record": True,
        "evaluation_output": str(output_json.resolve()),
        "audit_output": str(audit_json.resolve()),
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_count:
        raise RuntimeError(
            f"Direct baseline failed for {failed_count} record(s). Outputs were saved for audit; "
            "rerun the same command with resume enabled to retry failed records."
        )
    return predictions
