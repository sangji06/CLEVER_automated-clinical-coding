from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

from clever.phase2_code_retriever.pipeline import KCDEvaluator


def _load_phase2_output(phase2_output_path: str) -> Tuple[Dict[str, List[str]], List[dict], Dict[str, List[dict]]]:
    with open(phase2_output_path, "r", encoding="utf-8") as f:
        phase2_data = json.load(f)

    pred_dict = {}
    second_output_detail = []
    entries_by_document = {}

    for data_name, entries in phase2_data.items():
        codes = []
        normalized_entries = []
        for entry in entries:
            code = entry.get("kcd_code", "")
            if code:
                codes.append(code)
            normalized = {
                "data_name": data_name,
                "term": entry.get("term", ""),
                "refined_code": code,
                "description": entry.get("kcd_description", ""),
            }
            second_output_detail.append(normalized)
            normalized_entries.append(normalized)
        pred_dict[data_name] = codes
        entries_by_document[data_name] = normalized_entries

    return pred_dict, second_output_detail, entries_by_document


def _build_detail_data(entries_by_document: Dict[str, List[dict]], gt_codes_full: Dict[str, List[tuple]]) -> List[dict]:
    """Build a code-level detail sheet from saved Phase 2 output and GT.

    The original combined script produced richer details while retrieval and GT
    were in memory together. In the separated public workflow, this recreates
    the key TP/FP/FN detail rows without re-running FAISS retrieval.
    """
    detail_data = []
    all_document_ids = set(entries_by_document.keys()) | set(gt_codes_full.keys())

    for data_name in sorted(all_document_ids):
        model_entries = entries_by_document.get(data_name, [])
        gt_entries = gt_codes_full.get(data_name, [])

        model_by_code = defaultdict(list)
        gt_by_code = defaultdict(list)

        for item in model_entries:
            code = item.get("refined_code", "")
            if code:
                model_by_code[code].append(item)

        for term, code, description in gt_entries:
            if code:
                gt_by_code[code].append({
                    "term": term,
                    "code": code,
                    "description": description,
                })

        all_codes = set(model_by_code.keys()) | set(gt_by_code.keys())
        for code in sorted(all_codes):
            model_items = model_by_code.get(code, [])
            gt_items = gt_by_code.get(code, [])
            match_count = min(len(model_items), len(gt_items))

            for idx in range(match_count):
                model_item = model_items[idx]
                gt_item = gt_items[idx]
                detail_data.append({
                    "data_name": data_name,
                    "model_term": model_item.get("term", ""),
                    "model_code": code,
                    "model_description": model_item.get("description", ""),
                    "gt_term": gt_item.get("term", ""),
                    "gt_code": code,
                    "gt_description": gt_item.get("description", ""),
                    "match_type": "Code_Match",
                    "is_refined_out": False,
                    "Eval_Result": "TP",
                })

            for model_item in model_items[match_count:]:
                detail_data.append({
                    "data_name": data_name,
                    "model_term": model_item.get("term", ""),
                    "model_code": code,
                    "model_description": model_item.get("description", ""),
                    "gt_term": "",
                    "gt_code": "",
                    "gt_description": "",
                    "match_type": "Model_Only",
                    "is_refined_out": False,
                    "Eval_Result": "FP",
                })

            for gt_item in gt_items[match_count:]:
                detail_data.append({
                    "data_name": data_name,
                    "model_term": "",
                    "model_code": "",
                    "model_description": "",
                    "gt_term": gt_item.get("term", ""),
                    "gt_code": code,
                    "gt_description": gt_item.get("description", ""),
                    "match_type": "GT_Only",
                    "is_refined_out": False,
                    "Eval_Result": "FN",
                })

    return detail_data


def run_phase2_pipeline_evaluation(phase2_output_path: str, gt_path: str, output_excel: str):
    """Evaluate saved Phase 2 output against GT using the original metric logic.

    This keeps execution and evaluation separate. It reuses the original
    KCDEvaluator.extract_gt_codes(), KCDEvaluator.evaluate_predictions(), and
    KCDEvaluator.save_results_to_excel() methods without re-running FAISS retrieval.
    """
    pred_dict, second_output_detail, entries_by_document = _load_phase2_output(phase2_output_path)

    evaluator = KCDEvaluator.__new__(KCDEvaluator)
    with open(gt_path, "r", encoding="utf-8") as f:
        evaluator.gt_data = json.load(f)

    gt_codes_full = evaluator.extract_gt_codes()
    gt_codes_dict = {
        data_name: [code for _, code, _ in info_list]
        for data_name, info_list in gt_codes_full.items()
    }

    detail_data = _build_detail_data(entries_by_document, gt_codes_full)
    evaluation_results = evaluator.evaluate_predictions(pred_dict, gt_codes_dict)
    evaluator.save_results_to_excel(
        first_output={},
        second_output=pred_dict,
        evaluation_results=evaluation_results,
        detail_data=detail_data,
        first_output_detail=[],
        second_output_detail=second_output_detail,
        output_path=output_excel,
    )
    return evaluation_results
