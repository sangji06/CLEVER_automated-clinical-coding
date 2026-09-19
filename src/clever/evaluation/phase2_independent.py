import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------
# 1. KCD 대분류 매핑 함수 (공통 사용)
# ----------------------------------------------------------------------------------
def get_kcd_chapter_info(code):
    """
    KCD 코드를 받아 (대분류 코드 범위, 장 명칭)을 반환
    """
    code = str(code).strip().upper()
    if not code or code == 'NAN':
        return ('Unknown', '알 수 없음')

    first_char = code[0]
    prefix = code[:3]

    if 'A' <= first_char <= 'B': return ('A00-B99', '특정 감염성 및 기생충성 질환')
    if first_char == 'C': return ('C00-D48', '신생물')
    if first_char == 'D':
        return ('C00-D48', '신생물') if prefix <= 'D48' else ('D50-D89', '혈액 및 조혈기관의 질환과 면역매커니즘을 침범한 특정 장애')
    if first_char == 'E': return ('E00-E90', '내분비, 영양 및 대사 질환')
    if first_char == 'F': return ('F00-F99', '정신 및 행동 장애')
    if first_char == 'G': return ('G00-G99', '신경계통의 질환')
    if first_char == 'H':
        return ('H00-H59', '눈 및 눈 부속기의 질환') if prefix <= 'H59' else ('H60-H95', '귀 및 유돌의 질환')
    if first_char == 'I': return ('I00-I99', '순환계통의 질환')
    if first_char == 'J': return ('J00-J99', '호흡계통의 질환')
    if first_char == 'K': return ('K00-K93', '소화계통의 질환')
    if first_char == 'L': return ('L00-L99', '피부 및 피하조직의 질환')
    if first_char == 'M': return ('M00-M99', '근골격계통 및 결합조직의 질환')
    if first_char == 'N': return ('N00-N99', '비뇨생식계통의 질환')
    if first_char == 'O': return ('O00-O99', '임신, 출산 및 산후기')
    if first_char == 'P': return ('P00-P96', '출생전후기에 기원한 특정 병태')
    if first_char == 'Q': return ('Q00-Q99', '선천기형, 변형 및 염색체 이상')
    if first_char == 'R': return ('R00-R99', '달리 분류되지 않은 증상, 징후와 임상 및 검사의 이상소견')
    if 'S' <= first_char <= 'T': return ('S00-T98', '손상, 중독 및 외인에 의한 특정 기타 결과')
    if 'V' <= first_char <= 'Y': return ('V01-Y98', '질병이환 및 사망의 외인')
    if first_char == 'Z': return ('Z00-Z99', '건강상태 및 보건서비스 접촉에 영향을 주는 요인')
    if first_char == 'U': return ('U00-U99', '특수목적 코드')

    return ('Others', '기타')

def analyze_all_results(input_file, output_file):
    # 1. 데이터 로드
    df = pd.read_csv(input_file)

    # 2. 필수 컬럼 전처리 (description 포함)
    # 처리할 컬럼 목록 정의
    cols_to_check = [
        'gt_code', 'model_code',
        'gt_term', 'model_term',
        'gt_description', 'model_description'
    ]

    for col in cols_to_check:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            # CSV에 해당 컬럼이 없으면 빈 값으로 생성 (에러 방지)
            df[col] = ""

    df = df.dropna(subset=['gt_code']) # 1차 안전장치

    # --------------------------------------------------------------------------
    # PART A. 기존 분석 로직 (Table 1, Table 2)
    # --------------------------------------------------------------------------

    # 매핑 정보 생성
    gt_mapped = df['gt_code'].apply(get_kcd_chapter_info)
    df['GT_Chapter_Range'] = [x[0] for x in gt_mapped]
    df['GT_Chapter_Desc'] = [x[1] for x in gt_mapped]

    # Unknown 데이터 제거
    unknown_count = len(df[df['GT_Chapter_Range'] == 'Unknown'])
    if unknown_count > 0:
        print(f"알림: 유효하지 않은 GT 코드(Unknown) {unknown_count}건을 분석에서 제외합니다.")
        df = df[df['GT_Chapter_Range'] != 'Unknown'].copy()

    model_mapped = df['model_code'].apply(get_kcd_chapter_info)
    df['Model_Chapter_Range'] = [x[0] for x in model_mapped]
    df['Model_Chapter_Desc'] = [x[1] for x in model_mapped]

    df['Is_Correct'] = (df['model_code'] == df['gt_code'])

    # [Table 1] 대분류별 정확도
    table1 = df.groupby(['GT_Chapter_Range', 'GT_Chapter_Desc']).agg(
        Count=('gt_code', 'count'),
        Correct_Count=('Is_Correct', 'sum')
    ).reset_index()

    total_len = len(df)

    if total_len == 0:
        print("유효한 데이터가 0건입니다.")
        return

    table1['Accuracy (%)'] = (table1['Correct_Count'] / table1['Count']) * 100
    table1['Ratio (%)'] = (table1['Count'] / total_len) * 100
    table1 = table1.sort_values(by='Count', ascending=False)

    table1_final = table1[['GT_Chapter_Range', 'GT_Chapter_Desc', 'Count', 'Ratio (%)', 'Accuracy (%)']].copy()
    table1_final.columns = ['대분류 코드', '장(Chapters)', '빈도수 (건)', '비율 (%)', '정확도 (%)']

    # Total 행 추가
    total_correct = df['Is_Correct'].sum()
    total_accuracy = (total_correct / total_len) * 100

    total_row = pd.DataFrame([{
        '대분류 코드': 'Total',
        '장(Chapters)': '전체 (Overall)',
        '빈도수 (건)': total_len,
        '비율 (%)': 100.0,
        '정확도 (%)': total_accuracy
    }])
    table1_final = pd.concat([table1_final, total_row], ignore_index=True)
    table1_final['비율 (%)'] = table1_final['비율 (%)'].round(1)
    table1_final['정확도 (%)'] = table1_final['정확도 (%)'].round(1)

    # [Table 2] 세분화 수준별 정확도
    l1_acc = (df['GT_Chapter_Range'] == df['Model_Chapter_Range']).mean() * 100
    l2_acc = (df['model_code'].str[:3] == df['gt_code'].str[:3]).mean() * 100
    l3_acc = df['Is_Correct'].mean() * 100

    table2_data = [
        ['대분류 (Chapter)', '대분류 코드 범위(Range) 일치', l1_acc, 'S06.0 vs T01.9 (범위 일치)'],
        ['중분류 (Block)', '소수점 앞 3자리 일치', l2_acc, 'S06.0 vs S06.1'],
        ['세분류 (Full Code)', '전체 코드 완전 일치', l3_acc, 'S06.0 vs S06.0']
    ]
    table2_final = pd.DataFrame(table2_data, columns=['세분화 수준', '일치 판정 기준', '정확도 (%)', '일치 예시'])
    table2_final['정확도 (%)'] = table2_final['정확도 (%)'].round(1)

    # --------------------------------------------------------------------------
    # PART B. 신규 심층 오답 분석 (Error Analysis)
    # --------------------------------------------------------------------------

    # 오답 데이터 추출 (ACC = False)
    df_error = df[df['Is_Correct'] == False].copy()

    if len(df_error) > 0:
        # 중분류(앞3자리) 추출
        df_error['GT_Middle'] = df_error['gt_code'].str[:3]
        df_error['Model_Middle'] = df_error['model_code'].str[:3]

        # Type 정의
        mask_type_a = (df_error['GT_Middle'] == df_error['Model_Middle'])
        mask_type_b = (df_error['GT_Chapter_Range'] == df_error['Model_Chapter_Range']) & (~mask_type_a)
        mask_type_c = (df_error['GT_Chapter_Range'] != df_error['Model_Chapter_Range'])

        # [수정됨] 저장할 컬럼 및 순서 지정 (Chapter_Desc 제거, Description 추가)
        cols_order = [
            'model_term', 'model_code', 'model_description',
            'gt_term', 'gt_code', 'gt_description'
        ]

        df_type_a = df_error.loc[mask_type_a, cols_order].copy()
        df_type_b = df_error.loc[mask_type_b, cols_order].copy()
        df_type_c = df_error.loc[mask_type_c, cols_order].copy()

        # 오답 통계 요약
        total_errors = len(df_error)
        summary_data = [
            ['Type A (중분류 일치)', '소수점 앞 3자리는 맞춤 (예: K80 vs K80)', len(df_type_a), (len(df_type_a)/total_errors)*100],
            ['Type B (대분류 일치)', '같은 챕터 내 다른 코드 (예: K25 vs K92)', len(df_type_b), (len(df_type_b)/total_errors)*100],
            ['Type C (완전 불일치)', '아예 다른 챕터로 예측 (예: H vs G)', len(df_type_c), (len(df_type_c)/total_errors)*100],
            ['Total Errors', '전체 오답 수', total_errors, 100.0]
        ]
        df_error_summary = pd.DataFrame(summary_data, columns=['오답 유형', '설명', '건수', '비율(%)'])
        df_error_summary['비율(%)'] = df_error_summary['비율(%)'].round(1)
    else:
        df_error_summary = pd.DataFrame()
        df_type_a = pd.DataFrame()
        df_type_b = pd.DataFrame()
        df_type_c = pd.DataFrame()

    # --------------------------------------------------------------------------
    # PART C. 엑셀 저장
    # --------------------------------------------------------------------------
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 기존 통계
        table1_final.to_excel(writer, sheet_name='Table1_Chapter', index=False)
        table2_final.to_excel(writer, sheet_name='Table2_Level', index=False)

        # 오답 분석 (수정된 컬럼 순서 반영)
        df_error_summary.to_excel(writer, sheet_name='Error_Summary', index=False)
        df_type_a.to_excel(writer, sheet_name='Type_A_Middle_Match', index=False)
        df_type_b.to_excel(writer, sheet_name='Type_B_Chapter_Match', index=False)
        df_type_c.to_excel(writer, sheet_name='Type_C_Mismatch', index=False)

    print(f"분석 완료: '{output_file}'")
    print(" - Table 1, 2: 기존 통계 유지")
    print(" - Type A, B, C 시트: Description 추가 및 컬럼 순서 재정렬 완료 (Model -> GT)")

# ----------------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------------


def run_phase2_independent_evaluation(input_csv: str, output_xlsx: str):
    """Run the original Phase 2 independent evaluation."""
    return analyze_all_results(input_csv, output_xlsx)
