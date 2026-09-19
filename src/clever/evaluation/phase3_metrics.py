import json
import pandas as pd
from collections import Counter

def normalize_code(code):
    if code is None:
        return ""
    return str(code).strip().upper()


def run_phase3_evaluation(gt_path: str, model_output_path: str, output_xlsx: str):
    """Run the original cumulative Phase 3 evaluation."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    with open(model_output_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    detailed_rows = []

    # Partial Match(Main Code) 평가를 위한 전역 변수
    total_main_tp = 0
    total_main_fp = 0
    total_main_fn = 0

    # 2. 데이터 순회 및 정렬
    for gt_item in gt_data:
        raw_gt_id = gt_item.get('document_id', '')
        gt_id = raw_gt_id.rsplit('.', 1)[0] if '.' in raw_gt_id else raw_gt_id

        # (1) GT 정보 수집 (코드 기준)
        gt_info = {}
        gt_codes_list = [] # Main Code 평가용 리스트

        if 'tab3_final_codes' in gt_item and 'after_merge' in gt_item['tab3_final_codes']:
            for entry in gt_item['tab3_final_codes']['after_merge']:
                code = normalize_code(entry.get('KCD Code'))
                if not code: continue

                gt_codes_list.append(code) # 리스트에 추가 (중복 허용)

                term = entry.get('Term', '')
                desc = entry.get('KCD Description', '')

                # 동일 코드가 여러 번 등장할 경우 용어(Term)를 합쳐서 보여줌
                if code not in gt_info:
                    gt_info[code] = {'Term': term, 'Description': desc}
                else:
                    if term and term not in gt_info[code]['Term']:
                        gt_info[code]['Term'] += f", {term}"

        # (2) 모델 예측 정보 수집 (코드 기준)
        model_info = {}
        model_codes_list = [] # Main Code 평가용 리스트

        if gt_id in model_data:
            for entry in model_data[gt_id]:
                code = normalize_code(entry.get('kcd_code'))
                if not code: continue

                model_codes_list.append(code) # 리스트에 추가

                term = entry.get('term', '')
                desc = entry.get('kcd_description', '')

                if code not in model_info:
                    model_info[code] = {'Term': term, 'Description': desc}
                else:
                     if term and term not in model_info[code]['Term']:
                        model_info[code]['Term'] += f", {term}"

        # ====================================================
        # [NEW] Partial Match (Main Code) 평가 로직
        # ====================================================
        # 각 코드를 앞 3자리(Main Code)로 변환
        gt_main_codes = [c.split('.')[0] for c in gt_codes_list]
        model_main_codes = [c.split('.')[0] for c in model_codes_list]

        # Counter를 이용해 개수 기반 매칭 (예: C34가 GT에 2개, Model에 1개면 -> TP=1, FN=1)
        gt_main_counter = Counter(gt_main_codes)
        model_main_counter = Counter(model_main_codes)

        all_main_keys = set(gt_main_counter.keys()) | set(model_main_counter.keys())

        doc_main_tp = 0
        doc_main_fp = 0
        doc_main_fn = 0

        for m_code in all_main_keys:
            g_cnt = gt_main_counter[m_code]
            m_cnt = model_main_counter[m_code]

            doc_main_tp += min(g_cnt, m_cnt)
            doc_main_fp += max(0, m_cnt - g_cnt)
            doc_main_fn += max(0, g_cnt - m_cnt)

        # 전역 변수에 누적
        total_main_tp += doc_main_tp
        total_main_fp += doc_main_fp
        total_main_fn += doc_main_fn

        # 상세 결과 시트에 표시하기 위해 Main Code 존재 여부 Set 생성
        gt_main_set = set(gt_main_codes)
        model_main_set = set(model_main_codes)

        # ====================================================
        # (3) Exact Match 상세 비교 행 생성
        # ====================================================
        all_codes = set(gt_info.keys()).union(set(model_info.keys()))

        for code in sorted(list(all_codes)):
            # GT 데이터 가져오기
            gt_term = gt_info[code]['Term'] if code in gt_info else ""
            gt_desc = gt_info[code]['Description'] if code in gt_info else ""
            gt_code_display = code if code in gt_info else ""

            # Model 데이터 가져오기
            model_term = model_info[code]['Term'] if code in model_info else ""
            model_desc = model_info[code]['Description'] if code in model_info else ""
            model_code_display = code if code in model_info else ""

            # Exact Match 결과 판정
            if code in gt_info and code in model_info:
                status = 'TP'
            elif code in gt_info and code not in model_info:
                status = 'FN'
            elif code not in gt_info and code in model_info:
                status = 'FP'
            else:
                status = 'Error'

            # Main Code 매칭 여부 (참고용)
            # 현재 행의 코드(Exact)가 Main Code 기준으로는 상대방에 존재하는지 확인
            main_code = code.split('.')[0]
            main_match_status = "No"

            if status == 'TP':
                main_match_status = "Yes (Exact)"
            elif status == 'FN':
                # GT에는 있는데 모델이 Exact로는 못 맞춤. 하지만 Main Code는 맞췄나?
                if main_code in model_main_set:
                    main_match_status = "Yes (Partial)"
            elif status == 'FP':
                # GT에 없는데 모델이 예측함. 하지만 GT에 같은 Main Code가 있나?
                if main_code in gt_main_set:
                    main_match_status = "Yes (Partial)"

            detailed_rows.append({
                'Document ID': raw_gt_id,
                'GT Term': gt_term,
                'GT Code': gt_code_display,
                'GT Description': gt_desc,
                'Model Term': model_term,
                'Model Code': model_code_display,
                'Model Description': model_desc,
                'Exact Result': status,          # 기존 Result -> Exact Result
                'Main Code': main_code,          # 추가
                'Main Code Matched?': main_match_status # 추가 (참고용)
            })

    # 3. 결과 저장
    df_detailed = pd.DataFrame(detailed_rows)

    # ----------------------------------------------------
    # Summary 계산 (Exact Match & Partial Match)
    # ----------------------------------------------------

    # 1) Exact Match Metrics
    exact_tp = len(df_detailed[df_detailed['Exact Result'] == 'TP'])
    exact_fp = len(df_detailed[df_detailed['Exact Result'] == 'FP'])
    exact_fn = len(df_detailed[df_detailed['Exact Result'] == 'FN'])

    exact_prec = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0
    exact_rec = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0
    exact_f1 = 2 * exact_prec * exact_rec / (exact_prec + exact_rec) if (exact_prec + exact_rec) > 0 else 0

    # 2) Partial Match Metrics (Counter로 계산된 값 사용)
    main_prec = total_main_tp / (total_main_tp + total_main_fp) if (total_main_tp + total_main_fp) > 0 else 0
    main_rec = total_main_tp / (total_main_tp + total_main_fn) if (total_main_tp + total_main_fn) > 0 else 0
    main_f1 = 2 * main_prec * main_rec / (main_prec + main_rec) if (main_prec + main_rec) > 0 else 0

    # 3) Summary DataFrame 생성
    summary_rows = [
        {'Evaluation Type': 'Exact Match', 'Metric': 'Total TP', 'Value': exact_tp},
        {'Evaluation Type': 'Exact Match', 'Metric': 'Total FP', 'Value': exact_fp},
        {'Evaluation Type': 'Exact Match', 'Metric': 'Total FN', 'Value': exact_fn},
        {'Evaluation Type': 'Exact Match', 'Metric': 'Precision', 'Value': round(exact_prec, 4)},
        {'Evaluation Type': 'Exact Match', 'Metric': 'Recall', 'Value': round(exact_rec, 4)},
        {'Evaluation Type': 'Exact Match', 'Metric': 'F1 Score', 'Value': round(exact_f1, 4)},

        {'Evaluation Type': 'Partial Match (Main Code)', 'Metric': 'Total TP', 'Value': total_main_tp},
        {'Evaluation Type': 'Partial Match (Main Code)', 'Metric': 'Total FP', 'Value': total_main_fp},
        {'Evaluation Type': 'Partial Match (Main Code)', 'Metric': 'Total FN', 'Value': total_main_fn},
        {'Evaluation Type': 'Partial Match (Main Code)', 'Metric': 'Precision', 'Value': round(main_prec, 4)},
        {'Evaluation Type': 'Partial Match (Main Code)', 'Metric': 'Recall', 'Value': round(main_rec, 4)},
        {'Evaluation Type': 'Partial Match (Main Code)', 'Metric': 'F1 Score', 'Value': round(main_f1, 4)},
    ]

    summary_df = pd.DataFrame(summary_rows)

    # 엑셀 파일 쓰기
    output_filename = output_xlsx
    with pd.ExcelWriter(output_filename) as writer:
        df_detailed.to_excel(writer, sheet_name='Detailed Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Complete. Check {output_filename}")
