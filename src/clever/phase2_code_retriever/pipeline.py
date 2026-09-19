import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from collections import defaultdict, Counter  # Counter 추가됨
import warnings
warnings.filterwarnings('ignore')

class KCDEvaluator:
    def __init__(self,
                 faiss_index_path: str,
                 model_output_path: str,
                 gt_path: Optional[str] = None):
        """
        KCD 코드 평가기 초기화
        """
        self.faiss_index_path = faiss_index_path
        self.model_output_path = model_output_path
        self.gt_path = gt_path

        # 임베딩 모델 로드
        print("임베딩 모델 로딩 중...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )

        # FAISS 인덱스 로드
        print("FAISS 인덱스 로딩 중...")
        self.faiss_vectorstore = FAISS.load_local(
            faiss_index_path,
            self.embeddings_model,
            allow_dangerous_deserialization=True
        )

        # 데이터 로드
        print("데이터 로딩 중...")
        with open(model_output_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        if gt_path is not None:
            with open(gt_path, 'r', encoding='utf-8') as f:
                self.gt_data = json.load(f)
        else:
            self.gt_data = []

        # GT 데이터를 dictionary로 변환 (document_id를 키로)
        self.gt_dict = {}
        for item in self.gt_data:
            doc_id = item['document_id'].replace('.txt', '')
            self.gt_dict[doc_id] = item

    def convert_terms_to_kcd(self, terms: List[str]) -> List[Tuple[str, str, str]]:
        """
        Term들을 KCD 코드로 변환 (기존 코드 유지)
        """
        results = []

        for term in terms:
            if not term or term.strip() == "":
                continue

            try:
                # k=1로 가장 유사한 1개만 검색
                search_results = self.faiss_vectorstore.similarity_search(term, k=1)
                if search_results:

                    kcd_code_str = search_results[0].metadata.get('code', '')
                    description = search_results[0].page_content

                    if kcd_code_str:
                        individual_codes = kcd_code_str.split(',')
                        for code in individual_codes:
                            stripped_code = code.strip()
                            if stripped_code:
                                results.append((term, stripped_code, description))
            except Exception as e:
                print(f"Term '{term}' 변환 중 오류: {e}")
                continue

        return results

    def refine_kcd_codes(self, code_info_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        KCD 코드 정제 규칙 적용 (기존 코드 유지)
        """
        if not code_info_list:
            return []

        refined_list = []
        seen_codes = set()
        seen_a09_entries = set()

        for term, code, desc in code_info_list:
            if code == 'A09.9':
                term_key = term.strip().lower()
                entry_key = (term_key, code)

                if entry_key not in seen_a09_entries:
                    refined_list.append((term, code, desc))
                    seen_a09_entries.add(entry_key)

            elif code not in seen_codes:
                refined_list.append((term, code, desc))
                seen_codes.add(code)

        main_code_groups = defaultdict(list)
        for term, code, desc in refined_list:
            if '.' in code:
                main_code = code.split('.')[0]
                main_code_groups[main_code].append((term, code, desc))
            else:
                main_code_groups[code].append((term, code, desc))

        final_list = []

        for main_code, group_items in main_code_groups.items():
            if len(group_items) == 1:
                final_list.extend(group_items)
                continue

            codes_only = [item[1] for item in group_items]
            refined_codes = self.apply_refinement_rules(codes_only, main_code)

            refined_codes_set = set(refined_codes)

            for item in group_items:
                if item[1] in refined_codes_set:
                    final_list.append(item)

        return final_list

    def apply_refinement_rules(self, codes: List[str], main_code: str) -> List[str]:
        """
        그룹화된 코드에 정제 규칙 적용 (기존 코드 유지)
        """
        if len(codes) <= 1:
            return codes

        if main_code == 'A09':
            return codes

        if main_code == 'R10' and 'R10.49' in codes:
            if len(codes) > 1:
                codes = [c for c in codes if c != 'R10.49']

        if len(codes) <= 1:
            return codes

        if main_code == 'C34' and 'C34.99' in codes:
            if len(codes) > 1:
                codes = [c for c in codes if c != 'C34.99']

        if len(codes) <= 1:
            return codes

        code_info = []
        for code in codes:
            parts = code.split('.')
            if len(parts) == 2:
                sub_code = parts[1]
                code_info.append({
                    'code': code,
                    'sub_code': sub_code,
                    'length': len(code),
                    'has_9': '9' in sub_code
                })
            else:
                code_info.append({
                    'code': code,
                    'sub_code': '',
                    'length': len(code),
                    'has_9': False
                })

        if all(not c['has_9'] for c in code_info):
            return codes

        four_digit_codes = [c for c in code_info if c['length'] == 5 and '.' in c['code']]
        if four_digit_codes:
            five_digit_codes = [c for c in code_info if c['length'] == 6 and '.' in c['code']]
            nine_codes = [c for c in four_digit_codes if c['sub_code'] == '9']
            non_nine_codes = [c for c in four_digit_codes if c['sub_code'] != '9']

            if nine_codes and (non_nine_codes or five_digit_codes):
                codes = [c['code'] for c in non_nine_codes] + [c['code'] for c in five_digit_codes]
                code_info = [c for c in code_info if c['code'] in codes]
            elif nine_codes and not non_nine_codes and not five_digit_codes:
                return codes
            elif non_nine_codes:
                return codes

        five_digit_codes = [c for c in code_info if c['length'] == 6 and '.' in c['code']]
        if five_digit_codes:
            result_codes = []
            fourth_digit_groups = defaultdict(list)
            for c in five_digit_codes:
                if len(c['sub_code']) > 0:
                    fourth_digit = c['sub_code'][0]
                    fourth_digit_groups[fourth_digit].append(c)

            for fourth_digit, group in fourth_digit_groups.items():
                if len(group) == 1:
                    result_codes.append(group[0]['code'])
                    continue

                has_99 = any(c['sub_code'] == '99' for c in group)
                if has_99:
                    non_99_codes = [c['code'] for c in group if c['sub_code'] != '99']
                    if non_99_codes:
                        result_codes.extend(non_99_codes)
                    else:
                        result_codes.extend([c['code'] for c in group])
                else:
                    if len(group) > 1 and all(len(c['sub_code']) > 1 for c in group):
                        fifth_nine = [c for c in group if c['sub_code'][1] == '9']
                        fifth_non_nine = [c for c in group if c['sub_code'][1] != '9']

                        if fifth_nine and fifth_non_nine:
                            result_codes.extend([c['code'] for c in fifth_non_nine])
                        else:
                            result_codes.extend([c['code'] for c in group])
                    else:
                        result_codes.extend([c['code'] for c in group])

            if result_codes:
                final_codes = [c['code'] for c in code_info if c['length'] == 5 and '.' in c['code']]
                final_codes.extend(result_codes)
                return final_codes

        return codes

    def extract_gt_codes(self) -> Dict:
        """
        GT 데이터에서 KCD 코드 추출 (기존 코드 유지)
        """
        gt_codes = {}

        for item in self.gt_data:
            doc_id = item['document_id'].replace('.txt', '')

            if 'tab3_final_codes' in item and 'after_merge' in item['tab3_final_codes']:
                code_list = []
                for code_item in item['tab3_final_codes']['after_merge']:
                    term = code_item.get('Term', '')
                    code_str = code_item.get('KCD Code', '')
                    desc = code_item.get('KCD Description', '')

                    if code_str:
                        individual_codes = code_str.split(',')
                        for code in individual_codes:
                            stripped_code = code.strip()
                            if stripped_code:
                                code_list.append((term, stripped_code, desc))

                gt_codes[doc_id] = code_list

        return gt_codes

    def evaluate_predictions(self, pred_dict: Dict, gt_dict: Dict) -> Dict:
        """
        예측 코드와 GT 코드를 비교하여 평가 (수정됨: Main Code Match 추가)

        1. Exact Match (기존 방식): 코드가 완전히 일치해야 함 (C34.13 != C34.19)
        2. Main Code Match (신규 방식): 앞 3자리만 일치하면 됨 (C34 == C34)
        """
        # 1. Exact Match용 집계 변수
        exact_tp = 0; exact_fp = 0; exact_fn = 0

        # 2. Main Code Match용 집계 변수
        main_tp = 0; main_fp = 0; main_fn = 0

        # 그룹별 메트릭 초기화
        group_metrics_exact = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        group_metrics_main = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        # 문서별(doc_level) 스코어 저장용
        doc_level_metrics = {}

        common_keys = set(pred_dict.keys()).intersection(set(gt_dict.keys()))

        for data_name in common_keys:
            parts = data_name.split('_')
            group = parts[1] if len(parts) > 1 else 'Unknown'

            pred_list = pred_dict[data_name]
            gt_list = gt_dict[data_name]

            # ==========================================
            # 1. Exact Match Evaluation (기존 로직)
            # ==========================================
            pred_a09_count = pred_list.count('A09.9')
            gt_a09_count = gt_list.count('A09.9')

            pred_set = set(c for c in pred_list if c != 'A09.9')
            gt_set = set(c for c in gt_list if c != 'A09.9')

            tp_a09 = min(pred_a09_count, gt_a09_count)
            fp_a09 = max(0, pred_a09_count - gt_a09_count)
            fn_a09 = max(0, gt_a09_count - pred_a09_count)

            tp_other = len(pred_set.intersection(gt_set))
            fp_other = len(pred_set - gt_set)
            fn_other = len(gt_set - pred_set)

            cur_exact_tp = tp_a09 + tp_other
            cur_exact_fp = fp_a09 + fp_other
            cur_exact_fn = fn_a09 + fn_other

            # Global Accumulation (Exact)
            exact_tp += cur_exact_tp
            exact_fp += cur_exact_fp
            exact_fn += cur_exact_fn

            # Group Accumulation (Exact)
            group_metrics_exact[group]['tp'] += cur_exact_tp
            group_metrics_exact[group]['fp'] += cur_exact_fp
            group_metrics_exact[group]['fn'] += cur_exact_fn

            # ==========================================
            # 2. Main Code Match Evaluation (신규 로직)
            # ==========================================
            # 모든 코드를 앞 3자리로 변환 (예: C34.19 -> C34)
            pred_main_codes = [c.split('.')[0] for c in pred_list]
            gt_main_codes = [c.split('.')[0] for c in gt_list]

            # Counter를 사용하여 Multiset 개념으로 계산 (개수 기반 매칭)
            # 집합(Set)을 쓰면 C34.1, C34.2가 둘 다 C34가 되어 하나로 합쳐지는 문제 방지
            pred_counter = Counter(pred_main_codes)
            gt_counter = Counter(gt_main_codes)

            cur_main_tp = 0
            cur_main_fp = 0
            cur_main_fn = 0

            # 모든 등장한 Main Code에 대해 계산
            all_main_codes = set(pred_counter.keys()) | set(gt_counter.keys())

            for code in all_main_codes:
                p_count = pred_counter[code]
                g_count = gt_counter[code]

                cur_main_tp += min(p_count, g_count)
                cur_main_fp += max(0, p_count - g_count)
                cur_main_fn += max(0, g_count - p_count)

            # Global Accumulation (Main)
            main_tp += cur_main_tp
            main_fp += cur_main_fp
            main_fn += cur_main_fn

            # Group Accumulation (Main)
            group_metrics_main[group]['tp'] += cur_main_tp
            group_metrics_main[group]['fp'] += cur_main_fp
            group_metrics_main[group]['fn'] += cur_main_fn

            # ==========================================
            # 3. 문서별 메트릭 계산 및 저장
            # ==========================================
            doc_metrics_exact = self.calculate_metrics_from_counts(cur_exact_tp, cur_exact_fp, cur_exact_fn)
            doc_metrics_main = self.calculate_metrics_from_counts(cur_main_tp, cur_main_fp, cur_main_fn)

            doc_level_metrics[data_name] = {
                # Exact Match Scores (기존 컬럼명 유지 혹은 변경 -> 명확성을 위해 Exact 접두사 권장)
                'Exact_Precision': doc_metrics_exact['precision'],
                'Exact_Recall': doc_metrics_exact['recall'],
                'Exact_F1': doc_metrics_exact['f1'],

                # Main Code Match Scores (신규 추가)
                'Main_Precision': doc_metrics_main['precision'],
                'Main_Recall': doc_metrics_main['recall'],
                'Main_F1': doc_metrics_main['f1']
            }

        # -------------------------------------------------------
        # 전체 결과 집계
        # -------------------------------------------------------

        # 1. Overall Metrics
        overall_exact = self.calculate_metrics_from_counts(exact_tp, exact_fp, exact_fn)
        overall_main = self.calculate_metrics_from_counts(main_tp, main_fp, main_fn)

        overall_results = {
            'Exact_Match': overall_exact,
            'Main_Code_Match': overall_main
        }

        # 2. Group Metrics
        # Exact와 Main 결과를 하나의 딕셔너리 구조로 병합하거나 분리해서 저장
        # 여기서는 Group 이름 아래에 두 가지 평가 결과를 flattening 해서 저장
        group_results = {}
        all_groups = set(group_metrics_exact.keys()) | set(group_metrics_main.keys())

        for group in all_groups:
            # Exact
            e_counts = group_metrics_exact[group]
            e_metrics = self.calculate_metrics_from_counts(e_counts['tp'], e_counts['fp'], e_counts['fn'])

            # Main
            m_counts = group_metrics_main[group]
            m_metrics = self.calculate_metrics_from_counts(m_counts['tp'], m_counts['fp'], m_counts['fn'])

            # Flattened Dictionary for Excel
            group_results[group] = {
                'Exact_Precision': e_metrics['precision'],
                'Exact_Recall': e_metrics['recall'],
                'Exact_F1': e_metrics['f1'],
                'Main_Precision': m_metrics['precision'],
                'Main_Recall': m_metrics['recall'],
                'Main_F1': m_metrics['f1'],
                'Exact_TP': e_counts['tp'],
                'Exact_FP': e_counts['fp'],
                'Exact_FN': e_counts['fn'],
                'Main_TP': m_counts['tp'],
                'Main_FP': m_counts['fp'],
                'Main_FN': m_counts['fn']
            }

        return {
            'overall': overall_results,
            'by_group': group_results,
            'doc_level': doc_level_metrics
        }

    def calculate_metrics_from_counts(self, tp: int, fp: int, fn: int) -> Dict:
        """
        TP, FP, FN으로부터 메트릭 계산
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1': round(f1, 2),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def process_all_data(self) -> Tuple[Dict, Dict, List, List, List]:
        """
        모든 데이터 처리 및 평가 (기존 로직 유지)
        """
        first_output = {}
        second_output = {}
        detail_data = []
        first_output_detail = []
        second_output_detail = []

        gt_codes_full = self.extract_gt_codes()

        for data_name, data in self.model_data.items():
            terms = []
            if 'step4_expansion' in data:
                expansion = data['step4_expansion']
                if isinstance(expansion, list):
                    for item in expansion:
                        if isinstance(item, dict) and 'term' in item:
                            term = item.get('term', '').strip()
                            if term:
                                terms.append(term)

            converted_results = self.convert_terms_to_kcd(terms)
            first_output[data_name] = [code for _, code, _ in converted_results]

            for term, code, desc in converted_results:
                first_output_detail.append({
                    'data_name': data_name,
                    'term': term,
                    'converted_code': code,
                    'description': desc
                })

            refined_results = self.refine_kcd_codes(converted_results)
            second_output[data_name] = [code for _, code, _ in refined_results]

            refined_pairs = set((item[0], item[1]) for item in refined_results)

            for term, code, desc in refined_results:
                second_output_detail.append({
                    'data_name': data_name,
                    'term': term,
                    'refined_code': code,
                    'description': desc
                })

            if data_name in gt_codes_full:
                gt_items = gt_codes_full[data_name]

                model_pairs_list = [(item[0].lower(), item[1], item) for item in converted_results]
                gt_pairs_list = [(item[0].lower(), item[1], item) for item in gt_items]

                model_pairs_dict = defaultdict(list)
                gt_pairs_dict = defaultdict(list)

                for term_lower, code, item in model_pairs_list:
                    model_pairs_dict[(term_lower, code)].append(item)

                for term_lower, code, item in gt_pairs_list:
                    gt_pairs_dict[(term_lower, code)].append(item)

                model_keys = set(model_pairs_dict.keys())
                gt_keys = set(gt_pairs_dict.keys())

                processed_model_keys = set()
                processed_gt_keys = set()

                perfect_matches = model_keys.intersection(gt_keys)
                for key in perfect_matches:
                    model_items = model_pairs_dict[key]
                    gt_items_match = gt_pairs_dict[key]

                    count = min(len(model_items), len(gt_items_match))

                    for i in range(count):
                        model_term, model_code, model_desc = model_items[i]
                        is_refined_out = (model_term, model_code) not in refined_pairs

                        if not is_refined_out:
                            gt_term, gt_code, gt_desc = gt_items_match[i]
                            detail_data.append({
                                'data_name': data_name,
                                'model_term': model_term,
                                'model_code': model_code,
                                'model_description': model_desc,
                                'gt_term': gt_term,
                                'gt_code': gt_code,
                                'gt_description': gt_desc,
                                'match_type': 'Perfect_Match',
                                'is_refined_out': False,
                                'Eval_Result': 'TP'
                            })
                            processed_model_keys.add(key)
                            processed_gt_keys.add(key)

                model_keys_remaining = model_keys - processed_model_keys
                gt_keys_remaining = gt_keys - processed_gt_keys

                model_code_dict = defaultdict(list)
                for key in model_keys_remaining:
                    code = key[1]
                    for item in model_pairs_dict[key]:
                        if (item[0], item[1]) in refined_pairs:
                            model_code_dict[code].append((item, key))

                gt_code_dict = defaultdict(list)
                for key in gt_keys_remaining:
                    code = key[1]
                    gt_code_dict[code].extend([(item, key) for item in gt_pairs_dict[key]])

                common_codes = set(model_code_dict.keys()).intersection(set(gt_code_dict.keys()))

                for code in common_codes:
                    model_items_with_keys = model_code_dict[code]
                    gt_items_with_keys = gt_code_dict[code]

                    count = min(len(model_items_with_keys), len(gt_items_with_keys))

                    for i in range(count):
                        model_item, model_key = model_items_with_keys[i]
                        gt_item, gt_key = gt_items_with_keys[i]

                        model_term, model_code, model_desc = model_item
                        gt_term, gt_code, gt_desc = gt_item

                        detail_data.append({
                            'data_name': data_name,
                            'model_term': model_term,
                            'model_code': model_code,
                            'model_description': model_desc,
                            'gt_term': gt_term,
                            'gt_code': gt_code,
                            'gt_description': gt_desc,
                            'match_type': 'Code_Match_Term_Differ',
                            'is_refined_out': False,
                            'Eval_Result': 'TP'
                        })
                        processed_model_keys.add(model_key)
                        processed_gt_keys.add(gt_key)

                perfect_matches_refined = (model_keys.intersection(gt_keys)) - processed_model_keys
                for key in perfect_matches_refined:
                    if key in processed_gt_keys: continue

                    model_items = model_pairs_dict[key]
                    gt_items_match = gt_pairs_dict[key]

                    count = min(len(model_items), len(gt_items_match))

                    for i in range(count):
                        model_term, model_code, model_desc = model_items[i]
                        gt_term, gt_code, gt_desc = gt_items_match[i]
                        detail_data.append({
                            'data_name': data_name,
                            'model_term': model_term,
                            'model_code': model_code,
                            'model_description': model_desc,
                            'gt_term': gt_term,
                            'gt_code': gt_code,
                            'gt_description': gt_desc,
                            'match_type': 'Perfect_Match',
                            'is_refined_out': True,
                            'Eval_Result': 'FN (Refined TP)'
                        })
                        processed_model_keys.add(key)
                        processed_gt_keys.add(key)

                model_keys_remaining = model_keys - processed_model_keys
                gt_keys_remaining = gt_keys - processed_gt_keys

                model_code_dict_refined = defaultdict(list)
                for key in model_keys_remaining:
                    code = key[1]
                    for item in model_pairs_dict[key]:
                        if (item[0], item[1]) not in refined_pairs:
                            model_code_dict_refined[code].append((item, key))

                gt_code_dict_refined = defaultdict(list)
                for key in gt_keys_remaining:
                    code = key[1]
                    gt_code_dict_refined[code].extend([(item, key) for item in gt_pairs_dict[key]])

                common_codes_refined = set(model_code_dict_refined.keys()).intersection(set(gt_code_dict_refined.keys()))

                for code in common_codes_refined:
                    model_items_with_keys = model_code_dict_refined[code]
                    gt_items_with_keys = gt_code_dict_refined[code]

                    count = min(len(model_items_with_keys), len(gt_items_with_keys))

                    for i in range(count):
                        model_item, model_key = model_items_with_keys[i]
                        gt_item, gt_key = gt_items_with_keys[i]

                        model_term, model_code, model_desc = model_item
                        gt_term, gt_code, gt_desc = gt_item

                        detail_data.append({
                            'data_name': data_name,
                            'model_term': model_term,
                            'model_code': model_code,
                            'model_description': model_desc,
                            'gt_term': gt_term,
                            'gt_code': gt_code,
                            'gt_description': gt_desc,
                            'match_type': 'Code_Match_Term_Differ',
                            'is_refined_out': True,
                            'Eval_Result': 'FN (Refined TP)'
                        })
                        processed_model_keys.add(model_key)
                        processed_gt_keys.add(gt_key)

                model_keys_remaining = model_keys - processed_model_keys
                gt_keys_remaining = gt_keys - processed_gt_keys

                model_term_dict = defaultdict(list)
                for key in model_keys_remaining:
                    term_lower = key[0]
                    for item in model_pairs_dict[key]:
                        model_term_dict[term_lower].append((item, key))

                gt_term_dict = defaultdict(list)
                for key in gt_keys_remaining:
                    term_lower = key[0]
                    for item in gt_pairs_dict[key]:
                        gt_term_dict[term_lower].append((item, key))

                common_terms = set(model_term_dict.keys()).intersection(set(gt_term_dict.keys()))

                for term in common_terms:
                    model_items_with_keys = model_term_dict[term]
                    gt_items_with_keys = gt_term_dict[term]

                    count = min(len(model_items_with_keys), len(gt_items_with_keys))

                    for i in range(count):
                        model_item, model_key = model_items_with_keys[i]
                        gt_item, gt_key = gt_items_with_keys[i]

                        if model_key in processed_model_keys or gt_key in processed_gt_keys:
                            continue

                        model_term, model_code, model_desc = model_item
                        gt_term, gt_code, gt_desc = gt_item

                        is_refined_out = (model_term, model_code) not in refined_pairs
                        eval_result = 'FP/FN (Term Match)'

                        detail_data.append({
                            'data_name': data_name,
                            'model_term': model_term,
                            'model_code': model_code,
                            'model_description': model_desc,
                            'gt_term': gt_term,
                            'gt_code': gt_code,
                            'gt_description': gt_desc,
                            'match_type': 'Term_match_code_differ',
                            'is_refined_out': is_refined_out,
                            'Eval_Result': eval_result
                        })

                        processed_model_keys.add(model_key)
                        processed_gt_keys.add(gt_key)

                model_keys_remaining = model_keys - processed_model_keys
                for key in model_keys_remaining:
                    for item in model_pairs_dict[key]:
                        model_term, model_code, model_desc = item
                        is_refined_out = (model_term, model_code) not in refined_pairs

                        eval_result = 'FP' if not is_refined_out else 'Corrected (FP)'

                        detail_data.append({
                            'data_name': data_name,
                            'model_term': model_term,
                            'model_code': model_code,
                            'model_description': model_desc,
                            'gt_term': '',
                            'gt_code': '',
                            'gt_description': '',
                            'match_type': 'Model_Only',
                            'is_refined_out': is_refined_out,
                            'Eval_Result': eval_result
                        })

                gt_keys_remaining = gt_keys - processed_gt_keys
                for key in gt_keys_remaining:
                    for item in gt_pairs_dict[key]:
                        gt_term, gt_code, gt_desc = item

                        detail_data.append({
                            'data_name': data_name,
                            'model_term': '',
                            'model_code': '',
                            'model_description': '',
                            'gt_term': gt_term,
                            'gt_code': gt_code,
                            'gt_description': gt_desc,
                            'match_type': 'GT_Only',
                            'is_refined_out': False,
                            'Eval_Result': 'FN'
                        })
            else:
                for conv_term, conv_code, conv_desc in converted_results:
                    is_refined_out = (conv_term, conv_code) not in refined_pairs

                    eval_result = 'FP' if not is_refined_out else 'Corrected (FP)'

                    detail_data.append({
                        'data_name': data_name,
                        'model_term': conv_term,
                        'model_code': conv_code,
                        'model_description': conv_desc,
                        'gt_term': '',
                        'gt_code': '',
                        'gt_description': '',
                        'match_type': 'Model_Only',
                        'is_refined_out': is_refined_out,
                        'Eval_Result': eval_result
                    })

        return first_output, second_output, detail_data, first_output_detail, second_output_detail

    def save_results_to_excel(self,
                              first_output: Dict,
                              second_output: Dict,
                              evaluation_results: Dict,
                              detail_data: List,
                              first_output_detail: List,
                              second_output_detail: List,
                              output_path: str):
        """
        결과를 엑셀 파일로 저장 (수정됨: 2가지 기준 결과 저장)
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 전반적인 스코어 (Exact Match / Main Code Match 구분)
            # 딕셔너리를 DataFrame으로 변환할 때 행(row)으로 쌓이도록 처리
            overall_data = []
            for eval_type, metrics in evaluation_results['overall'].items():
                row = metrics.copy()
                row['Evaluation_Type'] = eval_type
                overall_data.append(row)

            # 'Evaluation_Type'을 맨 앞으로 보냄
            overall_df = pd.DataFrame(overall_data)
            cols = ['Evaluation_Type'] + [c for c in overall_df.columns if c != 'Evaluation_Type']
            overall_df = overall_df[cols]
            overall_df.to_excel(writer, sheet_name='Overall_Scores', index=False)

            # 2. 그룹별 스코어
            if evaluation_results['by_group']:
                group_df = pd.DataFrame(evaluation_results['by_group']).T
                group_df.reset_index(inplace=True)
                group_df.rename(columns={'index': 'Group'}, inplace=True)
                group_df.to_excel(writer, sheet_name='Group_Scores', index=False)

            # 3. 각 데이터별 상세 정보
            if detail_data:
                detail_df = pd.DataFrame(detail_data)

                if 'doc_level' in evaluation_results:
                    try:
                        doc_metrics_df = pd.DataFrame.from_dict(evaluation_results['doc_level'], orient='index')
                        doc_metrics_df.reset_index(inplace=True)
                        doc_metrics_df.rename(columns={'index': 'data_name'}, inplace=True)

                        detail_df = pd.merge(detail_df, doc_metrics_df, on='data_name', how='left')
                    except Exception as e:
                        print(f"Warning: data_name별 스코어 병합 중 오류 발생: {e}")

                detail_df.to_excel(writer, sheet_name='Detail_Data', index=False)

            # 4. 첫 번째 출력
            if first_output_detail:
                first_df = pd.DataFrame(first_output_detail)
                first_df.to_excel(writer, sheet_name='First_Output', index=False)

            # 5. 두 번째 출력
            if second_output_detail:
                second_df = pd.DataFrame(second_output_detail)
                second_df.to_excel(writer, sheet_name='Second_Output', index=False)

        print(f"결과가 {output_path}에 저장되었습니다.")

    def save_results_to_json(self, second_output_detail: List, output_path: str):
        """
        정제된 결과를 JSON 파일로 저장
        """
        json_output = defaultdict(list)

        for item in second_output_detail:
            data_name = item['data_name']

            entry = {
                "term": item['term'],
                "kcd_code": item['refined_code'],
                "kcd_description": item['description']
            }

            json_output[data_name].append(entry)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, ensure_ascii=False, indent=4)
            print(f"JSON 결과가 {output_path}에 저장되었습니다.")
        except Exception as e:
            print(f"JSON 저장 중 오류 발생: {e}")

    def run(self, excel_output_path: str = 'kcd_evaluation_results.xlsx', json_output_path: str = 'kcd_results.json'):
        """
        전체 평가 프로세스 실행
        """
        print("=" * 50)
        print("KCD 평가 시작")
        print("=" * 50)

        print("\n데이터 처리 중...")
        first_output, second_output, detail_data, first_output_detail, second_output_detail = self.process_all_data()

        gt_codes_full = self.extract_gt_codes()

        gt_codes_dict = {}
        for data_name, info_list in gt_codes_full.items():
            gt_codes_dict[data_name] = [code for _, code, _ in info_list]

        print("\n평가 수행 중...")
        # Exact 및 Main Code 평가 실행
        evaluation_results = self.evaluate_predictions(second_output, gt_codes_dict)

        print("\n" + "=" * 50)
        print("평가 결과")
        print("=" * 50)

        print("\n[전반적인 스코어 (Exact Match)]")
        for metric, value in evaluation_results['overall']['Exact_Match'].items():
            print(f"{metric}: {value}")

        print("\n[전반적인 스코어 (Main Code Match - 앞 3자리)]")
        for metric, value in evaluation_results['overall']['Main_Code_Match'].items():
            print(f"{metric}: {value}")

        print("\n[그룹별 스코어]")
        for group, metrics in evaluation_results['by_group'].items():
            print(f"\n{group}:")
            # 대표적으로 F1 스코어만 출력 (너무 길어지는 것 방지)
            print(f"  Exact F1: {metrics.get('Exact_F1', 0)}")
            print(f"  Main F1 : {metrics.get('Main_F1', 0)}")

        print("\n엑셀 파일 저장 중...")
        self.save_results_to_excel(
            first_output,
            second_output,
            evaluation_results,
            detail_data,
            first_output_detail,
            second_output_detail,
            excel_output_path
        )

        print("\nJSON 파일 저장 중...")
        self.save_results_to_json(second_output_detail, json_output_path)

        print("\n" + "=" * 50)
        print("평가 완료!")
        print("=" * 50)

        return first_output, second_output, evaluation_results


def run_phase2_retrieval_only(faiss_index_path: str, model_output_path: str, json_output_path: str):
    """Run Phase 2 retrieval/code refinement without ground-truth evaluation.

    Original change from Step2_Evaluation_pipeline.py: gt_path is optional so
    public users can generate Phase 2 output without private GT labels.
    """
    evaluator = KCDEvaluator(
        faiss_index_path=faiss_index_path,
        model_output_path=model_output_path,
        gt_path=None,
    )
    _, second_output, _, _, second_output_detail = evaluator.process_all_data()
    evaluator.save_results_to_json(second_output_detail, json_output_path)
    return second_output
