"""Original-style LLM entity extraction for CLEVER Phase 1 (Guided Extractor)."""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from clever.phase1_guided_extractor.prompts import DEFAULT_FEW_SHOTS, DEFAULT_PROMPTS


def initialize_bedrock_client(
    region_name: str = "us-west-2",
    read_timeout: int = 300,
    connect_timeout: int = 120,
):
    """Initialize an AWS Bedrock Runtime client, matching the original script."""
    config = Config(
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        retries={"max_attempts": 3},
    )
    return boto3.client(
        "bedrock-runtime",
        region_name=region_name,
        config=config,
    )


def load_medical_records_from_folder(
    folder_path: str | Path,
    *,
    preserve_filesystem_order: bool = False,
) -> tuple[list[str], list[str]]:
    """Load .txt records from a directory in the original filename/record format.

    ``preserve_filesystem_order=True`` reproduces the legacy experiment's
    ``os.listdir`` request order. The public/default path remains sorted.
    """
    folder_path = Path(folder_path)
    records: list[str] = []
    filenames: list[str] = []
    if preserve_filesystem_order:
        paths = [folder_path / name for name in os.listdir(folder_path) if name.endswith(".txt")]
    else:
        paths = sorted(folder_path.glob("*.txt"))
    for path in paths:
        records.append(path.read_text(encoding="utf-8"))
        filenames.append(path.name)
    return filenames, records


def to_source_text_like_streamlit(raw_content: str) -> str:
    """Create the SOURCE_TEXT used for start/end index calculation."""
    try:
        data_list = ast.literal_eval(raw_content)
        parts = []
        if isinstance(data_list, list):
            for item in data_list:
                if isinstance(item, dict):
                    for _, value in item.items():
                        parts.append(str(value) + "\n\n")
        source_text = "".join(parts).strip()
        return source_text if source_text else raw_content
    except Exception:
        return raw_content


def create_context_text(raw_content: str) -> str:
    """Create the FULL_CONTEXT_RECORD text used by the original script."""
    try:
        data_list = ast.literal_eval(raw_content)
        parts = []
        if isinstance(data_list, list):
            for item in data_list:
                if isinstance(item, dict):
                    for key, value in item.items():
                        parts.append(f"### {key}\n")
                        parts.append(f"{str(value)}\n\n")
        formatted_text = "".join(parts).strip()
        return formatted_text if formatted_text else raw_content
    except Exception:
        return raw_content


def parse_bedrock_output(
    response_text: str,
    *,
    allow_python_literal: bool = True,
) -> Any:
    """Safely parse the last JSON object/list from an LLM response."""
    if not isinstance(response_text, str) or not response_text.strip():
        return []

    text = response_text.strip()
    fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    candidates = fence_re.findall(text) or [text]

    def try_json(blob: str):
        try:
            return json.loads(blob)
        except Exception:
            if allow_python_literal:
                try:
                    return ast.literal_eval(blob)
                except Exception:
                    pass
            return None

    for blob in reversed(candidates):
        obj = try_json(blob)
        if obj is not None:
            return obj

    def extract_last_json_chunk(blob: str):
        last_idx = max(blob.rfind("["), blob.rfind("{"))
        if last_idx == -1:
            return None
        opener = blob[last_idx]
        closer = {"{": "}", "[": "]"}[opener]
        stack = []
        for idx in range(last_idx, len(blob)):
            char = blob[idx]
            if char == opener:
                stack.append(char)
            elif char == closer:
                stack.pop()
                if not stack:
                    return blob[last_idx : idx + 1]
        return None

    for blob in reversed(candidates):
        chunk = extract_last_json_chunk(blob)
        if chunk:
            obj = try_json(chunk)
            if obj is not None:
                return obj

    return []


def realign_simple_indices(parsed_items: Any, source_text: str) -> list[dict[str, Any]]:
    """Realign Step 1-3 flat entity spans against SOURCE_TEXT."""
    if not isinstance(parsed_items, list):
        return []

    fixed = []
    cursor = 0
    for item in parsed_items:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity", "")).strip()
        if not entity:
            continue

        pattern = r"\s+".join(re.escape(part) for part in entity.split())
        match = re.search(pattern, source_text[cursor:], flags=re.MULTILINE)
        if match:
            start = cursor + match.start()
            end = cursor + match.end()
        else:
            match_all = re.search(pattern, source_text, flags=re.MULTILINE)
            if not match_all:
                continue
            start = match_all.start()
            end = match_all.end()

        fixed.append({
            "entity": entity,
            "label": item.get("label", "Symptom"),
            "start": start,
            "end": end,
        })
        cursor = max(cursor, end)

    return fixed


def add_ids_to_original_entities(parsed_items: Any) -> list[dict[str, Any]]:
    """Add sequential ids to nested original_entities, as in the original script."""
    if not isinstance(parsed_items, list):
        return []

    id_counter = 0
    for item in parsed_items:
        if isinstance(item, dict) and isinstance(item.get("original_entities"), list):
            for entity in item["original_entities"]:
                if isinstance(entity, dict):
                    entity["id"] = id_counter
                    id_counter += 1
    return parsed_items


def realign_nested_indices_to_source(parsed_items: Any, source_text: str) -> list[dict[str, Any]]:
    """Realign nested Step 4 original_entities against SOURCE_TEXT."""
    if not isinstance(parsed_items, list):
        return []

    realigned_items = []
    for item in parsed_items:
        if not isinstance(item, dict):
            continue
        new_item = item.copy()
        new_original_entities = []
        entity_cursor_map: dict[str, int] = {}

        for entity in item.get("original_entities", []):
            if not isinstance(entity, dict):
                continue
            entity_text = str(entity.get("entity", "")).strip()
            if not entity_text:
                new_original_entities.append(entity)
                continue

            cursor = entity_cursor_map.get(entity_text, 0)
            pattern = r"\s+".join(re.escape(part) for part in entity_text.split())
            match = re.search(pattern, source_text[cursor:], flags=re.MULTILINE)
            if match:
                start = cursor + match.start()
                end = cursor + match.end()
                entity_cursor_map[entity_text] = end
            else:
                match_all = re.search(pattern, source_text, flags=re.MULTILINE)
                if not match_all:
                    new_original_entities.append(entity)
                    continue
                start = match_all.start()
                end = match_all.end()
                entity_cursor_map[entity_text] = end

            updated = entity.copy()
            updated["start"] = start
            updated["end"] = end
            new_original_entities.append(updated)

        new_item["original_entities"] = new_original_entities
        realigned_items.append(new_item)

    return realigned_items


def apply_expansion(
    step3_results: Any,
    step4_output: Any,
    *,
    legacy_exact: bool = False,
) -> list[dict[str, Any]]:
    """Use Step 4 output when valid; otherwise preserve Step 3 terms."""
    if legacy_exact and not step4_output:
        return [
            {"term": entity.get("entity", ""), "original_entities": [entity]}
            for entity in step3_results
            if isinstance(entity, dict)
        ]

    if isinstance(step4_output, list) and all(isinstance(item, dict) and "term" in item for item in step4_output):
        return step4_output

    if not isinstance(step3_results, list):
        return []

    return [
        {"term": entity.get("entity", ""), "original_entities": [entity]}
        for entity in step3_results
        if isinstance(entity, dict)
    ]


def call_bedrock(
    client,
    model_id: str,
    system_message: str,
    context_text: str,
    source_text: str,
    previous_result_json: str = "",
    few_shot_examples: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
    legacy_exact: bool = False,
) -> str:
    """Call Bedrock Converse API using the original prompt assembly pattern."""
    index_scope = "*only*" if legacy_exact else "only"
    if not previous_result_json:
        user_prompt = (
            f"{few_shot_examples}\n\n"
            f"---\n"
            f"Here is the full medical record for context. Read this to find the answers:\n\n"
            f"### FULL_CONTEXT_RECORD ###\n{context_text}\n\n"
            f"---\n"
            f"CRITICAL_RULE: After finding the answers, you MUST calculate all 'start' and 'end' indices based {index_scope} on the following SOURCE_TEXT. Do NOT use the FULL_CONTEXT_RECORD for indexing.\n\n"
            f"### SOURCE_TEXT (for index calculation) ###\n{source_text}\n\n"
            f"Output:"
        )
    else:
        user_prompt = (
            f"{few_shot_examples}\n\n"
            f"---\nHere is the task for the current step.\n\n"
            f"Here is the full medical record for context. Read this to find the answers:\n\n"
            f"### FULL_CONTEXT_RECORD ###\n{context_text}\n\n"
            f"---\n"
            f"Here are the results from the previous step:\n"
            f"PREVIOUS_STEP_RESULTS:\n{previous_result_json}\n\n"
            f"CRITICAL_RULE: You MUST calculate all 'start' and 'end' indices based {index_scope} on the following SOURCE_TEXT. Do NOT use the FULL_CONTEXT_RECORD for indexing.\n\n"
            f"### SOURCE_TEXT (for index calculation) ###\n{source_text}\n\n"
            f"Output:"
        )

    combined = (
        f"{system_message}\n\n{user_prompt}\n\n"
        f"IMPORTANT: Calculate all start/end indices based on SOURCE_TEXT, not the dictionary format.\n"
        f"Return ONLY JSON. No prose, no markdown fences."
    )

    messages = [{"role": "user", "content": [{"text": combined}]}]
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


class GuidedExtractor:
    """Run the original four-step Phase 1 guided extraction pipeline."""

    def __init__(
        self,
        client,
        model_id: str,
        prompts: dict[str, str] | None = None,
        few_shots: dict[str, str] | None = None,
        legacy_exact: bool = False,
        deterministic_step2_passthrough: bool = False,
    ):
        self.client = client
        self.model_id = model_id
        self.prompts = prompts or DEFAULT_PROMPTS
        self.few_shots = few_shots or DEFAULT_FEW_SHOTS
        self.legacy_exact = legacy_exact
        self.deterministic_step2_passthrough = deterministic_step2_passthrough

    def process_record(self, record: str, filename: str = "") -> dict[str, Any]:
        """Process one raw record string with the original source/context split."""
        results: dict[str, Any] = {
            "filename": filename,
            "status": "processing",
            "step1_extraction": None,
            "step2_elaboration": None,
            "step3_filtering": None,
            "step4_expansion": None,
            "error": None,
        }

        try:
            source_text = to_source_text_like_streamlit(record)
            context_text = create_context_text(record)

            step1_text = call_bedrock(
                self.client,
                self.model_id,
                self.prompts["step1"],
                context_text,
                source_text,
                "",
                self.few_shots["step1"],
                legacy_exact=self.legacy_exact,
            )
            step1 = realign_simple_indices(
                parse_bedrock_output(
                    step1_text,
                    allow_python_literal=not self.legacy_exact,
                ),
                source_text,
            )
            results["step1_extraction"] = step1
            current_result_json = json.dumps(step1, indent=2, ensure_ascii=False)

            if self.deterministic_step2_passthrough:
                # Control 1 removes the guideline-driven elaboration operation.
                # Copying in code guarantees that the neutralized step cannot
                # accidentally add, drop, or rewrite an entity.
                step2 = copy.deepcopy(step1)
            else:
                step2_text = call_bedrock(
                    self.client,
                    self.model_id,
                    self.prompts["step2"],
                    context_text,
                    source_text,
                    current_result_json,
                    self.few_shots["step2"],
                    legacy_exact=self.legacy_exact,
                )
                step2 = realign_simple_indices(
                    parse_bedrock_output(
                        step2_text,
                        allow_python_literal=not self.legacy_exact,
                    ),
                    source_text,
                )
            results["step2_elaboration"] = step2
            current_result_json = json.dumps(step2, indent=2, ensure_ascii=False)

            step3_text = call_bedrock(
                self.client,
                self.model_id,
                self.prompts["step3"],
                context_text,
                source_text,
                current_result_json,
                self.few_shots["step3"],
                legacy_exact=self.legacy_exact,
            )
            step3 = realign_simple_indices(
                parse_bedrock_output(
                    step3_text,
                    allow_python_literal=not self.legacy_exact,
                ),
                source_text,
            )
            results["step3_filtering"] = step3

            step4_text = call_bedrock(
                self.client,
                self.model_id,
                self.prompts["step4"],
                context_text,
                source_text,
                json.dumps(step3) if self.legacy_exact else json.dumps(step3, ensure_ascii=False),
                self.few_shots["step4"],
                legacy_exact=self.legacy_exact,
            )
            step4 = apply_expansion(
                step3,
                parse_bedrock_output(
                    step4_text,
                    allow_python_literal=not self.legacy_exact,
                ),
                legacy_exact=self.legacy_exact,
            )
            final_with_ids = add_ids_to_original_entities(step4)
            results["step4_expansion"] = realign_nested_indices_to_source(final_with_ids, source_text)
            results["status"] = "completed"
            return results
        except Exception as error:
            results["status"] = "failed"
            results["error"] = str(error)
            return results

    def process_records(
        self,
        filenames: list[str],
        records: list[str],
        max_workers: int = 1,
    ) -> dict[str, dict[str, Any]]:
        """Process multiple records; max_workers=1 keeps deterministic local runs."""
        output: dict[str, dict[str, Any]] = {}
        if max_workers <= 1:
            for filename, record in zip(filenames, records):
                output[Path(filename).stem] = self.process_record(record, filename)
            return output

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_record, record, filename): filename
                for filename, record in zip(filenames, records)
            }
            for future in as_completed(futures):
                filename = futures[future]
                output[Path(filename).stem] = future.result()
        return output


def run_extraction_dir(
    input_dir: str | Path,
    output_json: str | Path,
    model_id: str,
    region_name: str = "us-west-2",
    max_workers: int = 1,
) -> None:
    """Run Phase 1 extraction for all .txt records in a directory."""
    client = initialize_bedrock_client(region_name=region_name)
    extractor = GuidedExtractor(client=client, model_id=model_id)
    filenames, records = load_medical_records_from_folder(input_dir)
    results = extractor.process_records(filenames, records, max_workers=max_workers)

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
