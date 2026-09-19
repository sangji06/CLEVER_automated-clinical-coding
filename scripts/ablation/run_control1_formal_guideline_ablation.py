"""Run matched full and formal-guideline-ablation Phase 1 conditions.

The full condition uses the governed experimental prompt and few-shot package.
Control 1 uses the same source cases with only formal-guideline instructions
and associated demonstration content removed. Patient-derived examples are
supplied only from authorized local files and are never imported into the
public package.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path


# PyTorch/SapBERT and FAISS may load different OpenMP runtimes on Apple Silicon.
# Keeping the native libraries single-threaded prevents a segmentation fault
# during FAISS similarity search without changing the retrieved distances.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clever.ablation.control1_formal_guideline import (  # noqa: E402
    build_control1_prompts,
)
from clever.phase1_guided_extractor.extractor import (  # noqa: E402
    GuidedExtractor,
    initialize_bedrock_client,
    load_medical_records_from_folder,
)
FEW_SHOT_VARIABLES = {
    "step1": "FEW_SHOT_EXAMPLES_STEP1",
    "step2": "FEW_SHOT_EXAMPLES_STEP2",
    "step3": "FEW_SHOT_EXAMPLES_STEP3",
    "step4": "FEW_SHOT_EXAMPLES_STEP4",
}
PROMPT_STEPS = ("step1", "step2", "step3", "step4")


def load_private_few_shots(path: Path) -> dict[str, str]:
    """Read four string assignments from a private Python file without executing it."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in FEW_SHOT_VARIABLES.values():
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise ValueError(f"{target.id} must be a string in {path}")
        assignments[target.id] = value

    missing = [name for name in FEW_SHOT_VARIABLES.values() if name not in assignments]
    if missing:
        raise ValueError(f"Missing few-shot variables in {path}: {', '.join(missing)}")
    return {step: assignments[name] for step, name in FEW_SHOT_VARIABLES.items()}


def load_original_prompts(path: Path) -> dict[str, str]:
    """Read the original four-item ``prompts`` dictionary without executing code."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates: list[dict[str, str]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "prompts":
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict):
            continue
        if set(value) == set(PROMPT_STEPS) and all(
            isinstance(value[step], str) for step in PROMPT_STEPS
        ):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one four-step prompts dictionary in {path}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched Phase 1 conditions for the formal-guideline ablation. "
            "Use --condition both for the paired experiment."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--full-prompts-file",
        required=True,
        type=Path,
        help="Original experiment script containing the four-step prompts dictionary.",
    )
    parser.add_argument(
        "--full-few-shots-file",
        type=Path,
        help=(
            "Optional validation copy of the original full-condition few-shots. "
            "For exact legacy reproduction, Full always uses the few-shots embedded "
            "in --full-prompts-file; a supplied copy must match them exactly."
        ),
    )
    parser.add_argument(
        "--control1-few-shots-file",
        type=Path,
        help="Private file containing the matched guideline-ablated few-shots.",
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-workers", default=2, type=int)
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument(
        "--inter-batch-delay-seconds",
        default=5.0,
        type=float,
        help="Legacy experiment waited 5 seconds between Phase 1 batches.",
    )
    parser.add_argument(
        "--run-downstream",
        action="store_true",
        help="Run the shared curated-KCD Phase 2 and Phase 3 after Phase 1.",
    )
    parser.add_argument(
        "--faiss-index",
        type=Path,
        help="Curated KCD SapBERT FAISS index directory; required with --run-downstream.",
    )
    parser.add_argument(
        "--rules-csv",
        type=Path,
        help="Phase 3 correction-rule CSV; required with --run-downstream.",
    )
    parser.add_argument(
        "--condition",
        choices=("full", "control1", "both"),
        default="both",
        help="Run the full prompt, Control 1 prompt, or both matched conditions.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to use existing batch files in the selected output directory.",
    )
    return parser.parse_args()


def load_completed_batches(batch_dir: Path) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for batch_path in sorted(batch_dir.glob("batch_*.json")):
        with batch_path.open("r", encoding="utf-8") as file:
            batch_data = json.load(file)
        if not isinstance(batch_data, dict):
            raise ValueError(f"Batch file must contain a JSON object: {batch_path}")
        merged.update(batch_data)
    return merged


def next_batch_number(batch_dir: Path) -> int:
    numbers = []
    for batch_path in batch_dir.glob("batch_*.json"):
        try:
            numbers.append(int(batch_path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def run_condition(
    *,
    condition: str,
    client,
    model_id: str,
    filenames: list[str],
    records: list[str],
    output_dir: Path,
    prompts: dict[str, str],
    few_shots: dict[str, str],
    max_workers: int,
    batch_size: int,
    resume: bool,
    inter_batch_delay_seconds: float,
) -> Path:
    condition_dir = output_dir / condition
    batch_dir = condition_dir / "batches"
    final_output = condition_dir / "phase1_predictions.json"
    batch_dir.mkdir(parents=True, exist_ok=True)

    existing_batches = sorted(batch_dir.glob("batch_*.json"))
    if existing_batches and not resume:
        raise ValueError(
            f"Existing batch files found in {batch_dir}. "
            "Choose a new --output-dir or omit --no-resume."
        )

    completed = load_completed_batches(batch_dir) if resume else {}
    pending = [
        (filename, record)
        for filename, record in zip(filenames, records)
        if Path(filename).stem not in completed
    ]

    extractor = GuidedExtractor(
        client=client,
        model_id=model_id,
        prompts=prompts,
        few_shots=few_shots,
        legacy_exact=True,
        deterministic_step2_passthrough=(condition == "control1"),
    )
    batch_number = next_batch_number(batch_dir)

    for start in range(0, len(pending), batch_size):
        batch_pairs = pending[start : start + batch_size]
        batch_filenames = [item[0] for item in batch_pairs]
        batch_records = [item[1] for item in batch_pairs]
        batch_results = extractor.process_records(
            batch_filenames,
            batch_records,
            max_workers=max_workers,
        )

        batch_path = batch_dir / f"batch_{batch_number:03d}.json"
        batch_path.write_text(
            json.dumps(batch_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        completed.update(batch_results)
        print(
            f"[{condition}] saved batch {batch_number:03d}: "
            f"{len(batch_results)} records"
        )
        batch_number += 1
        if start + batch_size < len(pending) and inter_batch_delay_seconds > 0:
            time.sleep(inter_batch_delay_seconds)

    final_output.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    completed_count = sum(
        1 for result in completed.values() if result.get("status") == "completed"
    )
    failed_count = sum(
        1 for result in completed.values() if result.get("status") == "failed"
    )
    print(
        f"[{condition}] saved {len(completed)} records to {final_output} "
        f"(completed={completed_count}, failed={failed_count})"
    )
    return final_output


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be greater than 0.")
    if args.inter_batch_delay_seconds < 0:
        raise ValueError("--inter-batch-delay-seconds cannot be negative.")
    if args.run_downstream and (args.faiss_index is None or args.rules_csv is None):
        raise ValueError(
            "--faiss-index and --rules-csv are required with --run-downstream."
        )

    conditions = ("full", "control1") if args.condition == "both" else (args.condition,)
    full_prompts = load_original_prompts(args.full_prompts_file)
    original_full_few_shots = load_private_few_shots(args.full_prompts_file)
    if args.full_few_shots_file is not None:
        supplied_full_few_shots = load_private_few_shots(args.full_few_shots_file)
        if supplied_full_few_shots != original_full_few_shots:
            raise ValueError(
                "--full-few-shots-file does not exactly match the few-shots embedded "
                "in --full-prompts-file. Omit this flag for exact A reproduction."
            )

    condition_few_shots = {}
    if "full" in conditions:
        condition_few_shots["full"] = original_full_few_shots
    if "control1" in conditions:
        if args.control1_few_shots_file is None:
            raise ValueError(
                "--control1-few-shots-file is required for the Control 1 condition."
            )
        condition_few_shots["control1"] = load_private_few_shots(
            args.control1_few_shots_file
        )

    condition_prompts = {
        "full": full_prompts,
        "control1": build_control1_prompts(full_prompts),
    }

    filenames, records = load_medical_records_from_folder(
        args.input_dir,
        preserve_filesystem_order=True,
    )
    if not filenames:
        raise ValueError(f"No .txt files found in {args.input_dir}")

    client = initialize_bedrock_client(region_name=args.region)
    def text_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    manifest = {
        "legacy_exact": True,
        "model_id": args.model_id,
        "region": args.region,
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "inter_batch_delay_seconds": args.inter_batch_delay_seconds,
        "step2_execution": {
            condition: (
                "deterministic_copy_of_step1"
                if condition == "control1"
                else "original_llm_elaboration"
            )
            for condition in conditions
        },
        "full_prompts_file": str(args.full_prompts_file.resolve()),
        "prompt_sha256": {
            condition: {
                step: text_hash(condition_prompts[condition][step])
                for step in PROMPT_STEPS
            }
            for condition in conditions
        },
        "few_shot_sha256": {
            condition: {
                step: text_hash(condition_few_shots[condition][step])
                for step in PROMPT_STEPS
            }
            for condition in conditions
        },
        "conditions": list(conditions),
        "record_count": len(filenames),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    phase1_outputs = {}
    for condition in conditions:
        phase1_outputs[condition] = run_condition(
            condition=condition,
            client=client,
            model_id=args.model_id,
            filenames=filenames,
            records=records,
            output_dir=args.output_dir,
            prompts=condition_prompts[condition],
            few_shots=condition_few_shots[condition],
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            inter_batch_delay_seconds=args.inter_batch_delay_seconds,
        )

    if args.run_downstream:
        from clever.ablation.runner import run_phase2_and_phase3

        for condition in conditions:
            condition_dir = args.output_dir / condition
            phase2_output = condition_dir / "phase2_predictions.json"
            phase3_output = condition_dir / "phase3_predictions.json"
            rule_log = condition_dir / "phase3_applied_rules.csv"
            run_phase2_and_phase3(
                phase1_output=phase1_outputs[condition],
                faiss_index=args.faiss_index,
                rules_csv=args.rules_csv,
                phase2_output=phase2_output,
                phase3_output=phase3_output,
                rule_log=rule_log,
                phase1_schema="guided",
            )
            print(
                f"[{condition}] saved downstream outputs to {condition_dir}"
            )


if __name__ == "__main__":
    main()
