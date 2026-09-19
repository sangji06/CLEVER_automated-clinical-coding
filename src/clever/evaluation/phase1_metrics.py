"""Original Phase 1 evaluation logic for CLEVER Guided Extractor.

This module is a public-release refactor of `Step1_Evaluation.py`. The original
metric logic is preserved; only the hard-coded execution block was replaced by
a reusable `run_phase1_evaluation` helper for command-line use.
"""

import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class MedicalEntityEvaluator:
    def __init__(self, model_path: str, gt_path: str):
        """
        Initialize evaluator with model results and ground truth
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)

        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)
            # Convert GT list to dict with document_id as key
            self.gt_data = {item['document_id'].replace('.txt', ''): item for item in gt_list}

    def normalize_text(self, text: str) -> str:
        """Normalize text to lowercase and strip whitespace"""
        if not text: return ""
        return text.lower().strip()

    def normalize_term(self, text: str) -> str:
        """
        Normalize text for term matching: lowercase and remove all whitespace.
        """
        if not text: return ""
        return "".join(text.lower().split())

    def calculate_overlap_ratio(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
        """Calculate overlap ratio between two spans"""
        start1, end1 = span1
        start2, end2 = span2

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_length = max(0, overlap_end - overlap_start)

        len1 = end1 - start1
        len2 = end2 - start2
        min_length = min(len1, len2)

        if min_length == 0:
            return 0.0

        return overlap_length / min_length

    def evaluate_stage1_1_span(self, doc_id: str) -> Dict:
        """Evaluate Stage 1.1 (Entity Extraction - Span Based)"""
        if doc_id not in self.model_data or doc_id not in self.gt_data:
            return None

        model_entities = self.model_data[doc_id].get('step3_filtering') or []
        gt_entities = self.gt_data[doc_id].get('tab1_annotations') or []

        # Normalize entities for span matching
        model_normalized = [
            {
                'entity': self.normalize_text(e.get('entity', '')),
                'start': e.get('start'),
                'end': e.get('end'),
                'original': e
            }
            for e in model_entities
        ]

        gt_normalized = [
            {
                'entity': self.normalize_text(e.get('entity', '')),
                'start': e.get('start'),
                'end': e.get('end'),
                'original': e
            }
            for e in gt_entities
        ]

        # Exact match
        exact_results = self._match_entities(model_normalized, gt_normalized, match_type='exact')

        # Overlap match (90% threshold)
        overlap_results = self._match_entities(model_normalized, gt_normalized, match_type='overlap')

        return {
            'exact': exact_results,
            'overlap': overlap_results
        }

    def _match_entities(self, model_entities: List[Dict], gt_entities: List[Dict],
                        match_type: str) -> Dict:
        """Match model entities with GT entities"""
        model_matched = set()
        gt_matched = set()
        matches = []
        tp_entities = []

        for i, model_ent in enumerate(model_entities):
            for j, gt_ent in enumerate(gt_entities):
                is_match = False
                if match_type == 'exact':
                    if (model_ent['start'] == gt_ent['start'] and
                        model_ent['end'] == gt_ent['end'] and
                        model_ent['entity'] == gt_ent['entity']):
                        is_match = True

                elif match_type == 'overlap':
                    overlap_ratio = self.calculate_overlap_ratio(
                        (model_ent['start'], model_ent['end']),
                        (gt_ent['start'], gt_ent['end'])
                    )
                    if overlap_ratio >= 0.9:
                        is_match = True

                if is_match:
                    model_matched.add(i)
                    gt_matched.add(j)
                    matches.append((i, j))
                    tp_entities.append(model_ent['original'])

        tp = len(model_matched)
        fp = len(model_entities) - len(model_matched)
        fn = len(gt_entities) - len(gt_matched)

        fp_entities = [model_entities[i]['original'] for i in range(len(model_entities))
                       if i not in model_matched]
        fn_entities = [gt_entities[j]['original'] for j in range(len(gt_entities))
                       if j not in gt_matched]

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tp_entities': tp_entities,
            'fp_entities': fp_entities,
            'fn_entities': fn_entities,
            'matches': matches
        }

    def evaluate_stage1_1_term(self, doc_id: str) -> Dict:
        """Evaluate Stage 1.1 (Entity Extraction - Term Based)"""
        if doc_id not in self.model_data or doc_id not in self.gt_data:
            return None

        model_entities = self.model_data[doc_id].get('step3_filtering') or []
        gt_entities = self.gt_data[doc_id].get('tab1_annotations') or []

        model_terms_unique = {}
        for entity_obj in model_entities:
            # Check both 'entity' and 'term' just in case
            raw_text = entity_obj.get('entity') or entity_obj.get('term', '')
            term_normalized = self.normalize_term(raw_text)
            if term_normalized and term_normalized not in model_terms_unique:
                model_terms_unique[term_normalized] = entity_obj

        gt_terms_unique = {}
        for entity_obj in gt_entities:
            raw_text = entity_obj.get('entity') or entity_obj.get('term', '')
            term_normalized = self.normalize_term(raw_text)
            if term_normalized and term_normalized not in gt_terms_unique:
                gt_terms_unique[term_normalized] = entity_obj

        matched_terms = set(model_terms_unique.keys()) & set(gt_terms_unique.keys())

        tp = len(matched_terms)
        fp = len(model_terms_unique) - tp
        fn = len(gt_terms_unique) - tp

        fp_terms = [{'term': t, 'data': model_terms_unique[t]}
                    for t in model_terms_unique.keys() if t not in matched_terms]
        fn_terms = [{'term': t, 'data': gt_terms_unique[t]}
                    for t in gt_terms_unique.keys() if t not in matched_terms]

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'fp_terms': fp_terms,
            'fn_terms': fn_terms,
            'matched_terms_set': matched_terms,
        }

    def evaluate_stage1_2_term(self, doc_id: str) -> Dict:
        """Evaluate Stage 1.2 (Entity Refining)"""
        if doc_id not in self.model_data or doc_id not in self.gt_data:
            return None

        model_terms = self.model_data[doc_id].get('step4_expansion') or []
        gt_terms = self.gt_data[doc_id].get('tab2_refined_terms') or []

        gt_valid_answer_set = set()
        gt_valid_to_group_map = {}
        gt_key_groups = set()

        for term_obj in gt_terms:
            gt_key_norm = self.normalize_term(term_obj['term'])
            if not gt_key_norm: continue

            gt_valid_answer_set.add(gt_key_norm)
            gt_key_groups.add(gt_key_norm)
            gt_valid_to_group_map[gt_key_norm] = gt_key_norm

            for entity in term_obj.get('original_entities', []):
                answer_term = entity.get('expanded_term')
                if not answer_term:
                    answer_term = entity.get('entity')

                if answer_term:
                    answer_norm = self.normalize_term(answer_term)
                    if answer_norm:
                        gt_valid_answer_set.add(answer_norm)
                        gt_valid_to_group_map[answer_norm] = gt_key_norm

        tp_terms = set()
        fp_terms_list = []
        model_terms_unique = {}

        for term_obj in model_terms:
            # Safe get for term or entity
            model_key_norm = self.normalize_term(term_obj.get('term') or term_obj.get('entity', ''))
            if not model_key_norm or model_key_norm in model_terms_unique:
                continue

            model_terms_unique[model_key_norm] = term_obj

            if model_key_norm in gt_valid_answer_set:
                is_tp = True
            else:
                is_tp = False

            if is_tp:
                tp_terms.add(model_key_norm)
            else:
                fp_terms_list.append({'term': model_key_norm, 'data': term_obj})

        tp = len(tp_terms)
        fp = len(fp_terms_list)

        covered_gt_groups = set()
        for model_tp in tp_terms:
            if model_tp in gt_valid_to_group_map:
                covered_gt_groups.add(gt_valid_to_group_map[model_tp])

        uncovered_gt_groups = gt_key_groups - covered_gt_groups
        fn = len(uncovered_gt_groups)

        fn_terms_list = []
        for uncovered_key in uncovered_gt_groups:
             original_gt_obj = next((t for t in gt_terms if self.normalize_term(t['term']) == uncovered_key), None)
             fn_terms_list.append({'term': uncovered_key, 'data': original_gt_obj})

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'fp_terms': fp_terms_list,
            'fn_terms': fn_terms_list,
            'matched_terms_set': tp_terms,
        }

    def evaluate_stage1_3_final_term(self, doc_id: str) -> Dict:
        """Evaluate Stage 1.3: Compare GT 'after_merge' terms vs Model 'step4_expansion'"""
        if doc_id not in self.model_data or doc_id not in self.gt_data:
            return None

        model_terms = self.model_data[doc_id].get('step4_expansion') or []
        gt_tab3 = self.gt_data[doc_id].get('tab3_final_codes') or {}
        gt_after_merge = gt_tab3.get('after_merge') or []

        model_terms_norm = set()
        for t in model_terms:
            norm = self.normalize_term(t.get('term') or t.get('entity', ''))
            if norm:
                model_terms_norm.add(norm)

        gt_terms_norm = set()
        for item in gt_after_merge:
            norm = self.normalize_term(item.get('Term', ''))
            if norm:
                gt_terms_norm.add(norm)

        matched_terms = model_terms_norm & gt_terms_norm

        tp = len(matched_terms)
        fp = len(model_terms_norm) - tp
        fn = len(gt_terms_norm) - tp

        fp_terms = [t for t in model_terms_norm if t not in matched_terms]
        fn_terms = [t for t in gt_terms_norm if t not in matched_terms]

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'matched_terms': matched_terms,
            'fp_terms': fp_terms,
            'fn_terms': fn_terms
        }

    # [NEW] 분석 기능: Stage 1.1 TP가 Stage 1.2에서 어떻게 변했는지 추적
    def analyze_transitions(self, doc_id: str) -> List[Dict]:
        """
        Analyze terms that were TPs in Stage 1.1 but became FP or FN in Stage 1.2
        """
        # 1. Stage 1.1에서 성공한(TP) Entity들 확보
        s1_results = self.evaluate_stage1_1_term(doc_id)
        if not s1_results: return []
        s1_tp_entities = s1_results['matched_terms_set'] # normalized text set

        # 2. Stage 1.2 결과 (특히 FP와 FN 리스트) 확보
        s2_results = self.evaluate_stage1_2_term(doc_id)
        if not s2_results: return []

        transition_rows = []

        # --- Case A: TP -> FP (잘못된 매핑) ---
        for fp_item in s2_results['fp_terms']:
            model_term = fp_item['term']
            model_data = fp_item['data']

            for source_ent in model_data.get('original_entities', []):
                ent_text = source_ent.get('entity') or source_ent.get('term', '')
                ent_norm = self.normalize_term(ent_text)

                if ent_norm in s1_tp_entities:
                    transition_rows.append({
                        'Document': doc_id,
                        'Transition Type': 'TP -> FP (Wrong Mapping)',
                        'Entity (Stage 1.1)': ent_text,
                        'Model Output (Stage 1.2)': model_term,
                        'GT Expected (Stage 1.2)': 'Check GT'
                    })

        # --- Case B: TP -> FN (소실됨) ---
        for fn_item in s2_results['fn_terms']:
            gt_term = fn_item['term']
            gt_data = fn_item['data']

            found_source_entity = None
            if gt_data:
                for source_ent in gt_data.get('original_entities', []):
                    ent_text = source_ent.get('entity') or source_ent.get('expanded_term', '')
                    ent_norm = self.normalize_term(ent_text)

                    if ent_norm in s1_tp_entities:
                        found_source_entity = ent_text
                        break

            if found_source_entity:
                transition_rows.append({
                    'Document': doc_id,
                    'Transition Type': 'TP -> FN (Failed to Map)',
                    'Entity (Stage 1.1)': found_source_entity,
                    'Model Output (Stage 1.2)': '(Missing)',
                    'GT Expected (Stage 1.2)': gt_term
                })

        return transition_rows

    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict:
        """Calculate Precision, Recall, F1"""
        if tp + fp == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1': round(f1, 2)
        }

    def evaluate_all_documents(self) -> Dict:
        """Evaluate all documents and aggregate results"""
        results = {
            'document_level': [],
            'group_level': defaultdict(lambda: {
                'stage1_1_span_exact': [],
                'stage1_1_span_overlap': [],
                'stage1_1_term': [],
                'stage1_2_term': [],
                'stage1_3_final_term': []
            }),
            'transitions': []
        }

        for doc_id in self.model_data.keys():
            if doc_id not in self.gt_data:
                continue

            try:
                parts = doc_id.split('_')
                group = parts[1] if len(parts) >= 3 else 'Unknown'
            except:
                group = 'Unknown'

            # --- Stage 1.1 Span ---
            stage1_1_span_results = self.evaluate_stage1_1_span(doc_id)
            if stage1_1_span_results:
                def format_span_list(entity_list):
                    return '; '.join([e.get('entity', '') for e in entity_list if e.get('entity')])

                # Exact
                exact_res = stage1_1_span_results['exact']
                exact_metrics = self.calculate_metrics(exact_res['tp'], exact_res['fp'], exact_res['fn'])
                results['document_level'].append({
                    'Document': doc_id, 'Group': group, 'Phase': 'Stage1.1 (Span_Exact)',
                    'Precision': exact_metrics['precision'], 'Recall': exact_metrics['recall'], 'F1': exact_metrics['f1'],
                    'TP': exact_res['tp'], 'FP': exact_res['fp'], 'FN': exact_res['fn'],
                    'TP_Terms': format_span_list(exact_res['tp_entities']),
                    'FP_Terms': format_span_list(exact_res['fp_entities']),
                    'FN_Terms': format_span_list(exact_res['fn_entities'])
                })

                # Overlap
                overlap_res = stage1_1_span_results['overlap']
                overlap_metrics = self.calculate_metrics(overlap_res['tp'], overlap_res['fp'], overlap_res['fn'])
                results['document_level'].append({
                    'Document': doc_id, 'Group': group, 'Phase': 'Stage1.1 (Span_Overlap)',
                    'Precision': overlap_metrics['precision'], 'Recall': overlap_metrics['recall'], 'F1': overlap_metrics['f1'],
                    'TP': overlap_res['tp'], 'FP': overlap_res['fp'], 'FN': overlap_res['fn'],
                    'TP_Terms': format_span_list(overlap_res['tp_entities']),
                    'FP_Terms': format_span_list(overlap_res['fp_entities']),
                    'FN_Terms': format_span_list(overlap_res['fn_entities'])
                })

                results['group_level'][group]['stage1_1_span_exact'].append({'tp': exact_res['tp'], 'fp': exact_res['fp'], 'fn': exact_res['fn']})
                results['group_level'][group]['stage1_1_span_overlap'].append({'tp': overlap_res['tp'], 'fp': overlap_res['fp'], 'fn': overlap_res['fn']})

            # --- Stage 1.1 Term ---
            stage1_1_term_results = self.evaluate_stage1_1_term(doc_id)
            if stage1_1_term_results:
                metrics = self.calculate_metrics(stage1_1_term_results['tp'], stage1_1_term_results['fp'], stage1_1_term_results['fn'])

                # [MODIFIED] 다시 채워넣도록 수정함
                tp_str = '; '.join(stage1_1_term_results['matched_terms_set'])
                fp_str = '; '.join([t['term'] for t in stage1_1_term_results['fp_terms']])
                fn_str = '; '.join([t['term'] for t in stage1_1_term_results['fn_terms']])

                results['document_level'].append({
                    'Document': doc_id, 'Group': group, 'Phase': 'Stage1.1 (Term)',
                    'Precision': metrics['precision'], 'Recall': metrics['recall'], 'F1': metrics['f1'],
                    'TP': stage1_1_term_results['tp'], 'FP': stage1_1_term_results['fp'], 'FN': stage1_1_term_results['fn'],
                    'TP_Terms': tp_str,
                    'FP_Terms': fp_str,
                    'FN_Terms': fn_str
                })
                results['group_level'][group]['stage1_1_term'].append({'tp': stage1_1_term_results['tp'], 'fp': stage1_1_term_results['fp'], 'fn': stage1_1_term_results['fn']})

            # --- Stage 1.2 Term ---
            stage1_2_term_results = self.evaluate_stage1_2_term(doc_id)
            if stage1_2_term_results:
                metrics = self.calculate_metrics(stage1_2_term_results['tp'], stage1_2_term_results['fp'], stage1_2_term_results['fn'])

                # [MODIFIED] 다시 채워넣도록 수정함
                tp_str = '; '.join(stage1_2_term_results['matched_terms_set'])
                fp_str = '; '.join([t['term'] for t in stage1_2_term_results['fp_terms']])
                fn_str = '; '.join([t['term'] for t in stage1_2_term_results['fn_terms']])

                results['document_level'].append({
                    'Document': doc_id, 'Group': group, 'Phase': 'Stage1.2 (Term)',
                    'Precision': metrics['precision'], 'Recall': metrics['recall'], 'F1': metrics['f1'],
                    'TP': stage1_2_term_results['tp'], 'FP': stage1_2_term_results['fp'], 'FN': stage1_2_term_results['fn'],
                    'TP_Terms': tp_str,
                    'FP_Terms': fp_str,
                    'FN_Terms': fn_str
                })
                results['group_level'][group]['stage1_2_term'].append({'tp': stage1_2_term_results['tp'], 'fp': stage1_2_term_results['fp'], 'fn': stage1_2_term_results['fn']})

            # --- Stage 1.3 Final Term ---
            stage1_3_results = self.evaluate_stage1_3_final_term(doc_id)
            if stage1_3_results:
                metrics = self.calculate_metrics(stage1_3_results['tp'], stage1_3_results['fp'], stage1_3_results['fn'])
                results['document_level'].append({
                    'Document': doc_id, 'Group': group, 'Phase': 'Stage1.3 (Final_Term)',
                    'Precision': metrics['precision'], 'Recall': metrics['recall'], 'F1': metrics['f1'],
                    'TP': stage1_3_results['tp'], 'FP': stage1_3_results['fp'], 'FN': stage1_3_results['fn'],
                    'TP_Terms': '; '.join(stage1_3_results['matched_terms']) if stage1_3_results['matched_terms'] else '',
                    'FP_Terms': '; '.join(stage1_3_results['fp_terms']) if stage1_3_results['fp_terms'] else '',
                    'FN_Terms': '; '.join(stage1_3_results['fn_terms']) if stage1_3_results['fn_terms'] else ''
                })
                results['group_level'][group]['stage1_3_final_term'].append({'tp': stage1_3_results['tp'], 'fp': stage1_3_results['fp'], 'fn': stage1_3_results['fn']})

            # --- Transition Analysis Run ---
            transitions = self.analyze_transitions(doc_id)
            if transitions:
                results['transitions'].extend(transitions)

        return results

    def aggregate_group_level(self, group_data: List[Dict]) -> Dict:
        """Calculate micro average for a group"""
        if not group_data:
            return {'micro': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}

        total_tp = sum(d['tp'] for d in group_data)
        total_fp = sum(d['fp'] for d in group_data)
        total_fn = sum(d['fn'] for d in group_data)
        micro_metrics = self.calculate_metrics(total_tp, total_fp, total_fn)

        return {'micro': micro_metrics}

    def generate_report(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate evaluation report with tables"""
        results = self.evaluate_all_documents()

        # Table 1: Document-level
        document_df = pd.DataFrame(results['document_level'])

        # Table 2: Group-level
        group_rows = []
        for group, data in results['group_level'].items():
            phases = [
                ('Stage1.1 (Span_Exact)', 'stage1_1_span_exact'),
                ('Stage1.1 (Span_Overlap)', 'stage1_1_span_overlap'),
                ('Stage1.1 (Term)', 'stage1_1_term'),
                ('Stage1.2 (Term)', 'stage1_2_term'),
                ('Stage1.3 (Final_Term)', 'stage1_3_final_term')
            ]

            for phase_name, key in phases:
                agg = self.aggregate_group_level(data[key])
                group_rows.append({
                    'Group': group,
                    'Phase': phase_name,
                    'Avg Type': 'Micro',
                    'Precision': agg['micro']['precision'],
                    'Recall': agg['micro']['recall'],
                    'F1': agg['micro']['f1']
                })

        group_df = pd.DataFrame(group_rows)

        # Table 3: Overall
        overall_data = defaultdict(list)
        for group_data in results['group_level'].values():
            overall_data['stage1_1_span_exact'].extend(group_data['stage1_1_span_exact'])
            overall_data['stage1_1_span_overlap'].extend(group_data['stage1_1_span_overlap'])
            overall_data['stage1_1_term'].extend(group_data['stage1_1_term'])
            overall_data['stage1_2_term'].extend(group_data['stage1_2_term'])
            overall_data['stage1_3_final_term'].extend(group_data['stage1_3_final_term'])

        overall_rows = []
        phases = [
            ('Stage1.1 (Span_Exact)', 'stage1_1_span_exact'),
            ('Stage1.1 (Span_Overlap)', 'stage1_1_span_overlap'),
            ('Stage1.1 (Term)', 'stage1_1_term'),
            ('Stage1.2 (Term)', 'stage1_2_term'),
            ('Stage1.3 (Final_Term)', 'stage1_3_final_term')
        ]

        for phase_name, key in phases:
            agg = self.aggregate_group_level(overall_data[key])
            overall_rows.append({
                'Phase': phase_name,
                'Avg Type': 'Micro',
                'Precision': agg['micro']['precision'],
                'Recall': agg['micro']['recall'],
                'F1': agg['micro']['f1']
            })

        overall_df = pd.DataFrame(overall_rows)

        # Table 4: Transition Analysis
        transition_df = pd.DataFrame(results['transitions'])

        return document_df, group_df, overall_df, transition_df


def run_phase1_evaluation(model_path: str, gt_path: str, output_xlsx: str) -> None:
    """Run the original Phase 1 evaluation and save the Excel report."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', None)

    evaluator = MedicalEntityEvaluator(
        model_path=model_path,
        gt_path=gt_path,
    )
    doc_results, group_results, overall_results, transition_results = evaluator.generate_report()

    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        doc_results.to_excel(writer, sheet_name='Document Results', index=False)
        group_results.to_excel(writer, sheet_name='Group Results (Micro)', index=False)
        overall_results.to_excel(writer, sheet_name='Overall Results (Micro)', index=False)

        if not transition_results.empty:
            transition_results.to_excel(writer, sheet_name='Transition Analysis', index=False)
        else:
            pd.DataFrame(
                columns=[
                    'Document',
                    'Transition Type',
                    'Entity (Stage 1.1)',
                    'Model Output (Stage 1.2)',
                    'GT Expected (Stage 1.2)',
                ]
            ).to_excel(writer, sheet_name='Transition Analysis', index=False)
