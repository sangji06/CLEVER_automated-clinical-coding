import streamlit as st
import streamlit_antd_components as sac
import math
import os
import json
from datetime import datetime
import re
import glob
import ast
import pandas as pd

# --- 페이지 기본 설정 ---
st.set_page_config(layout="wide")

# 헤더 크기 줄이는 CSS 주입
st.markdown("""
<style>
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.3rem !important; }
    /* 사이드바 글씨 크기 조정 */
    [data-testid="stSidebar"] h2 { font-size: 1.1rem !important; }
    [data-testid="stSidebar"] .stMarkdown { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 경로 및 상태 관리 파일 정의 ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
PROGRESS_FILE = "progress.json"

# --- 함수 정의 ---
LABEL_COLORS = {"Diagnosis": "rgba(255, 105, 97, 0.7)", "Symptom": "rgba(119, 221, 119, 0.7)", "Supportive": "rgba(174, 198, 207, 0.7)"}

TERM_TO_KCD_MAPPING = {
    'fever': {'code': 'R50.99', 'desc': 'Fever, unspecified'},
    'dyspnea': {'code': 'R06.0', 'desc': 'Dyspnoea'},
    'mild dyspnea': {'code': 'R06.0', 'desc': 'Dyspnoea'},
    'acute dyspnea': {'code': 'R06.0', 'desc': 'Dyspnoea'},
    'vomiting': {'code': 'R11.2', 'desc': 'Vomiting alone'},
    'vomit': {'code': 'R11.2', 'desc': 'Vomiting alone'},
    'nausea': {'code': 'R11.1', 'desc': 'Nausea alone'},
    'cough': {'code': 'R05', 'desc': 'Cough'},
    'sputum': {'code': 'R09.3', 'desc': 'Abnormal sputum'},
    'dizziness': {'code': 'R42', 'desc': 'Dizziness and giddiness'},
    'neutropenic fever': {'code': 'D70', 'desc': 'Neutropenia'},
    'headache': {'code': 'R51', 'desc': 'Headache'},
    'mild headache': {'code': 'R51', 'desc': 'Headache'},
    'chest pain': {'code': 'R07.4', 'desc': 'Chest pain, unspecified'},
    'chill': {'code': 'R68.8', 'desc': 'Chills (without fever)'},
    'chills': {'code': 'R68.8', 'desc': 'Chills (without fever)'},
    'chilling': {'code': 'R68.8', 'desc': 'Chills (without fever)'},
    'chilling sense': {'code': 'R68.8', 'desc': 'Chills (without fever)'},
    'influenza': {'code': 'J11.1', 'desc': 'Influenza'},
    'upper respiratory infection': {'code': 'J06.9', 'desc': 'Upper respiratory infection'},
    'sore throat': {'code': 'J02.9', 'desc': 'Sore throat(acute)'},
    'pneumonia': {'code': 'J18.9', 'desc': 'Pneumonia, unspecified'},
    'rhinorrhea': {'code': 'J34.8', 'desc': 'Other specified disorders of nose and nasal sinuses'},
    'chest discomfort': {'code': 'R09.88', 'desc': 'Other specified symptoms and signs involving the circulatory and respiratory systems'},
    'abdominal pain': {'code': 'R10.49', 'desc': 'Unspecified abdominal pain'},
    'acute gastroenteritis': {'code': 'A09.9', 'desc': 'Gastroenteritis and colitis of unspecified origin'},
    'diarrhea': {'code': 'A09.9', 'desc': 'Gastroenteritis and colitis of unspecified origin'},
    'appendicitis': {'code': 'K37', 'desc': 'Unspecified appendicitis'},
    'acute appendicitis': {'code': 'K35.8', 'desc': 'Acute appendicitis, other and unspecified'},
    'nausea with vomiting': {'code': 'R11.3', 'desc': 'Nausea with vomiting'},
    'abdominal pain, lower abdomen': {'code': 'R10.39', 'desc': 'Lower abdominal pain, unspecified'},
    'flank pain': {'code': 'R10.44', 'desc': 'Flank pain'},
    'flank pain, right': {'code': 'R10.44', 'desc': 'Flank pain'},
    'flank pain, left': {'code': 'R10.44', 'desc': 'Flank pain'},
    'left flank pain': {'code': 'R10.44', 'desc': 'Flank pain'},
    'right flank pain': {'code': 'R10.44', 'desc': 'Flank pain'},
    'both flank pain': {'code': 'R10.44', 'desc': 'Flank pain'},
    'epigastric pain': {'code': 'R10.12', 'desc': 'Epigastric pain'},
    'gastritis': {'code': 'K29.7', 'desc': 'Gastritis, unspecified'},
    'acute gastritis': {'code': 'K29.1', 'desc': 'Other acute gastritis'},
    'stroke': {'code': 'I64', 'desc': 'Stroke, not specified as haemorrhage or infarction'},
    'cancer pain': {'code': 'R52.9', 'desc': 'Pain, unspecified'},
    'abdominal pain, right upper quadrant': {'code': 'R10.10', 'desc': 'Right upper quadrant abdominal pain'},
    'right upper quadrant abdominal pain': {'code': 'R10.10', 'desc': 'Right upper quadrant abdominal pain'},
    'abdominal pain, left upper quadrant': {'code': 'R10.11', 'desc': 'Left upper quadrant abdominal pain'},
    'left upper quadrant abdominal pain': {'code': 'R10.11', 'desc': 'Left upper quadrant abdominal pain'},
    'abdominal pain, upper abdomen': {'code': 'R10.19', 'desc': 'Upper abdominal pain, unspecified'},
    'upper abdominal pain, unspecified': {'code': 'R10.19', 'desc': 'Upper abdominal pain, unspecified'},
    'abdominal pain, right lower quadrant': {'code': 'R10.30', 'desc': 'Right lower quadrant abdominal pain'},
    'right lower quadrant abdominal pain': {'code': 'R10.30', 'desc': 'Right lower quadrant abdominal pain'},
    'abdominal pain, left lower quadrant': {'code': 'R10.31', 'desc': 'Left lower quadrant abdominal pain'},
    'left lower quadrant abdominal pain': {'code': 'R10.31', 'desc': 'Left lower quadrant abdominal pain'},
    'periumbilical pain': {'code': 'R10.32', 'desc': 'Periumbilical pain'},
    'abdominal pain, periumbilical': {'code': 'R10.32', 'desc': 'Periumbilical pain'},
    'abdominal pain, lower abdomen': {'code': 'R10.39', 'desc': 'Lower abdominal pain, unspecified'},
    'hematuria': {'code': 'R31.8', 'desc': 'Other and unspecified hematuria'},
    'abdominal discomfort': {'code': 'R19.8', 'desc': 'Other specified symptoms and signs involving the digestive system and abdomen'},
    'spontaneous bacterial peritonitis': {'code': 'K65.8', 'desc': 'Spontaneous bacterial peritonitis[SBP]'},
    'urinary tract infection': {'code': 'N39.0', 'desc': 'Urinary tract infection, site not specified'},
    'dyspepsia': {'code': 'R10.19', 'desc': 'Dyspepsia'},
    'colitis': {'code': 'A09.9', 'desc': 'Gastroenteritis and colitis of unspecified origin'},
    'acute colitis': {'code': 'A09.9', 'desc': 'Gastroenteritis and colitis of unspecified origin'},
    'cholangitis': {'code': 'K83.0', 'desc': 'Cholangitis'},
    'hepatocellular carcinoma': {'code': 'C22.0', 'desc': 'Malignant neoplasm of liver cell carcinoma'},
    'viral infection': {'code': 'B34.9', 'desc': 'Viral infection, unspecified'},
    'back pain': {'code': 'M54.99', 'desc': 'Backache, site unspecified'},
    'low back pain': {'code': 'M54.59', 'desc': 'Low back pain, site unspecified'},
    'lower back pain': {'code': 'M54.59', 'desc': 'Low back pain, site unspecified'},
    'breast cancer': {'code': 'C50.99', 'desc': 'Malignant neoplasm of breast unspecified, unspecified side'},
    'breast cancer, right': {'code': 'C50.90', 'desc': 'Malignant neoplasm of breast unspecified, right'},
    'breast cancer, left': {'code': 'C50.91', 'desc': 'Malignant neoplasm of breast unspecified, left'},
    'general weakness': {'code': 'R53', 'desc': 'General weakness'},
    'acute pyelonephritis': {'code': 'N10', 'desc': 'Acute pyelonephritis'},
    'acute coronary syndrome': {'code': 'I24.9', 'desc': 'Acute ischaemic heart disease, unspecified'},
    'chronic kidney disease': {'code': 'N18.9', 'desc': 'Chronic kidney disease, unspecified'},
    'acute kidney injury': {'code': 'N17.9', 'desc': 'Acute renal failure, unspecified'},
    'vestibular neuritis': {'code': 'H81.2', 'desc': 'Vestibular neuronitis'},
    'gastroesophageal reflux disease': {'code': 'K21.9', 'desc': 'Gastro-oesophageal reflux disease without oesophagitis'},
    'constipation': {'code': 'K59.09', 'desc': 'Other and unspecified constipation'},
    'pleural effusion': {'code': 'J90', 'desc': 'Pleural effusion'},
    'pulmonary thromboembolism': {'code': 'I26.9', 'desc': 'Pulmonary embolism without mention of acute cor pulmonale'},
    'anemia': {'code': 'D64.9', 'desc': 'Anaemia, unspecified'},
    'cancer fever': {'code': 'R50.8', 'desc': 'Other specified fever'},
    'mild headache': {'code': 'R51', 'desc': 'Headache'},
    'skin rash': {'code': 'R21', 'desc': 'Rash and other nonspecific skin eruption'},
    'angina': {'code': 'I20.9', 'desc': 'Angina'},
    'dysuria': {'code': 'R30.0', 'desc': 'Dysuria'},
    'heart failure': {'code': 'I50.9', 'desc': 'Heart failure, unspecified'},
    'lung cancer': {'code': 'C34.99', 'desc': 'Malignant neoplasm of bronchus or lung, unspecified, unspecified side'},
    'watery diarrhea': {'code': 'A09.0', 'desc': 'Acute watery diarrhoea'},
    'myocardial infarction': {'code': 'I21.9', 'desc': 'Myocardial infarction (acute)'},
    'peptic ulcer disease': {'code': 'K27.9', 'desc': 'Unspecified as acute or chronic peptic ulcer, site unspecified without hemorrhage or perforation'},
    'acute pharyngotonsillitis': {'code': 'J06.8', 'desc': 'Other acute upper respiratory infections of multiple sites'},
    'chronic obstructive pulmonary disease': {'code': 'J44.99', 'desc': 'Chronic obstructive pulmonary disease, unspecified, unspecified'},
    'herniated intervertebral disc': {'code': 'M51.2', 'desc': 'Other specified intervertebral disc displacement'},
    'unstable angina': {'code': 'I20.0', 'desc': 'Unstable angina'},
    'benign paroxysmal positional vertigo': {'code': 'H81.1', 'desc': 'Benign paroxysmal positional vertigo'},
    'meningitis': {'code': 'G03.9', 'desc': 'Meningitis, unspecified'},
    'pelvic inflammatory disease': {'code': 'N73.9', 'desc': 'Female pelvic inflammatory disease, unspecified'},
    'palpitation': {'code': 'R00.2', 'desc': 'Palpitations'},
    'syncope': {'code': 'R55.8', 'desc': 'Other syncope and collapse'},
    'arrhythmia': {'code': 'I49.9', 'desc': 'Arrhythmia (cardiac)'},
    'ureter stone': {'code': 'N20.1', 'desc': 'Calculus of ureter'},
    'seizure': {'code': 'R56.8', 'desc': 'Seizure (convulsive)'},
    'abdominal distension': {'code': 'R14', 'desc': 'Abdominal distension, gaseous'},
    'ascite': {'code': 'R18', 'desc': 'Ascites'},
    'ascites': {'code': 'R18', 'desc': 'Ascites'},
    'ileus': {'code': 'K56.7', 'desc': 'Ileus, unspecified'},
    'orthostatic hypotension': {'code': 'I95.1', 'desc': 'Orthostatic hypotension'},
    'urinary frequency': {'code': 'R35.0', 'desc': 'Frequency of micturition'},
    'urinary retention': {'code': 'R33', 'desc': 'Urinary retention'},
    'myalgia': {'code': 'M79.198', 'desc': 'Other myalgia, site unspecified'},
    'anorexia': {'code': 'R63.0', 'desc': 'Anorexia'},
    'poor oral intake': {'code': 'R63.0', 'desc': 'Loss of appetite'},
    'melena': {'code': 'K92.1', 'desc': 'Melaena'},
    'hematochezia': {'code': 'K92.1', 'desc': 'Melaena'},
    'cellulitis': {'code': 'L03.9', 'desc': 'Cellulitis, unspecified'},
}

ABBREVIATION_MAPPING = {
    'uri': 'Upper Respiratory Infection',
    'uti': 'Urinary Tract Infection',
    'ckd': 'Chronic Kidney Disease',
    'apn': 'Acute Pyelonephritis',
    'nf': 'Neutropenic fever',
    'age': 'Acute Gastroenteritis',
    'pn': 'Pneumonia',
    'acs': 'Acute Coronary Syndrome',
    'aki': 'Acute Kidney Injury',
    'flu': 'Influenza',
    'hf': 'Heart failure',
    'vn': 'Vestibular neuritis',
    'sbp': 'Spontaneous Bacterial Peritonitis',
    'bppv': 'Benign paroxysmal positional vertigo',
    'pid': 'Pelvic inflammatory disease',
    'gerd': 'Gastroesophageal reflux disease',
    'gw': 'General weakness',
    'pte': 'Pulmonary thromboembolism',
    'mi': 'Myocardial infarction',
    'pud': 'Peptic Ulcer Disease',
    'ha': 'Headache',
    'apt': 'Acute pharyngotonsillitis',
    'copd': 'Chronic Obstructive Pulmonary Disease',
    'hivd': 'Herniated intervertebral disc',
    'ua': 'Unstable angina',
    'poi': 'poor oral intake',
    'lt.': 'left',
    'rt.': 'right',
    'abd pain': 'abdominal pain',
    'doe': 'Dyspnea on exertion',
    'rlq': 'right lower quadrant',
    'ruq': 'right upper quadrant',
    'llq': 'left lower quadrant',
    'resp. difficulty': 'Respiratory difficulty',
    '배가 아프다': 'abdominal pain',
    'abd. pain': 'abdominal pain',
}

MERGE_CODE_MAPPING = {
    frozenset(['R11.1', 'R11.2']): {
        'term': 'Nausea with Vomiting',
        'code': 'R11.3',
        'desc': 'Nausea with Vomiting'
    },
    frozenset(['R50.99', 'R68.8']): {
        'term': 'Fever with Chills',
        'code': 'R50.8',
        'desc': 'Fever with Chills'
    },
}

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

def save_progress(progress_data):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f: json.dump(progress_data, f, ensure_ascii=False, indent=4)

def save_all_tabs_to_file(document_id, source_text):
    """ 모든 탭의 현재 상태를 저장하는 통합 저장 함수 """
    if not document_id: return None
    
    # ✨ 수정: tab1_annotations에서 entity가 있는 것만 필터링
    tab1_annotations_raw = st.session_state.get("tab1_annotations", [])
    tab1_annotations_filtered = [
        anno for anno in tab1_annotations_raw 
        if anno.get("entity") and anno.get("entity").strip()  # entity가 있고 빈 문자열이 아닌 경우만
    ]
    
    # ✨✨✨ [요청사항 수정] data_to_save에 tab2_expanded_annotations 추가
    data_to_save = {
        "document_id": document_id,
        "tab1_annotations": tab1_annotations_filtered,
        "tab2_expanded_annotations": st.session_state.get("tab2_expanded_annotations", []), # 1단계 통과 항목
        "tab2_refined_terms": st.session_state.get("tab2_refined_terms", []), # 2단계 통과(최종 그룹) 항목
        "tab3_final_codes": st.session_state.get("tab3_final_codes", {})
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(document_id)[0]
    output_filename = f"{base_filename}_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        with open(output_path, "w", encoding='utf-8') as f: json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        return output_path
    except Exception as e:
        st.error(f"파일 저장 중 오류 발생: {e}"); return None

def preprocess_record(raw_content: str):
    display_parts, source_parts, source_to_display_map = [], [], []
    current_display_idx = 0
    try:
        data_list = ast.literal_eval(raw_content)
        if isinstance(data_list, list) and data_list:
            for item in data_list:
                if isinstance(item, dict):
                    for key, value in item.items():
                        key_str = f"### {key}\n"
                        display_parts.append(key_str); display_parts.append(str(value) + "\n\n")
                        source_parts.append(str(value) + "\n\n")
                        current_display_idx += len(key_str)
                        for _ in str(value) + "\n\n":
                            source_to_display_map.append(current_display_idx); current_display_idx += 1
            return "".join(display_parts).strip(), "".join(source_parts).strip(), source_to_display_map
    except (ValueError, SyntaxError):
        text_map = list(range(len(raw_content)))
        return raw_content, raw_content, text_map

def get_annotated_html(display_text, source_to_display_map, annotations):
    def escape_html(text): return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    valid_annotations = sorted([a for a in annotations if a.get("entity") and a.get("start", -1) != -1], key=lambda x: x["start"])
    
    parts = []
    last_display_end = 0
    
    for anno in valid_annotations:
        source_start, source_end, label = anno["start"], anno["end"], anno["label"]
        if source_start < 0 or source_end > len(source_to_display_map): continue
        
        display_start = source_to_display_map[source_start]
        display_end = source_to_display_map[source_end - 1] + 1 if source_end > 0 and source_end <= len(source_to_display_map) else display_start

        if display_start >= last_display_end:
            plain_text = display_text[last_display_end:display_start]
            parts.append(escape_html(plain_text))
            color = LABEL_COLORS.get(label, "#FFFFFF")
            annotated_text_html = escape_html(display_text[display_start:display_end])
            html_tag = (f'<span class="annotated-text" style="background-color: {color};">'
                        f'{annotated_text_html} <span class="annotated-label">{label}</span></span>')
            parts.append(html_tag)
            last_display_end = display_end
            
    if last_display_end < len(display_text):
        parts.append(escape_html(display_text[last_display_end:]))
    
    html_body = "".join(parts).replace('\n', '<br>')
    
    full_html = f"""
    <head>
        <style>
            body {{ font-family: sans-serif; font-size: 1rem; font-weight: normal; line-height: 1.8; }}
            h3 {{ font-size: 1.2rem !important; font-weight: bold !important; margin-top: 1.5em; margin-bottom: 0.5em; }}
            .annotated-text {{ padding: 0.2em 0.4em; margin: 0 0.2em; line-height: 1; border-radius: 0.35em; }}
            .annotated-label {{ font-size: 0.8em; font-weight: bold; color: black; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    """
    return full_html

def clean_progress_data():
    """progress.json에서 존재하지 않는 파일 정보 제거"""
    if not os.path.exists(PROGRESS_FILE):
        return 0
    
    progress_data = load_progress()
    try:
        existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    except FileNotFoundError:
        existing_files = []
    
    # 존재하는 파일만 남기기
    cleaned_data = {f: status for f, status in progress_data.items() if f in existing_files}
    
    # 변경사항이 있으면 저장
    if len(cleaned_data) != len(progress_data):
        save_progress(cleaned_data)
        return len(progress_data) - len(progress_data)  # 삭제된 항목 수 반환
    return 0

# ✨✨✨ 추가: 새 annotation의 고유 ID를 생성하는 함수
def get_next_annotation_id():
    """현재 annotations에서 사용 중인 ID 중 최댓값 + 1을 반환"""
    existing_ids = [anno.get('id', -1) for anno in st.session_state.get('tab1_annotations', [])]
    return max(existing_ids) + 1 if existing_ids else 0

# ✨✨✨ [요청사항 수정] handle_file_select 함수 수정
def handle_file_select(new_filename):
    old_filename = st.session_state.get("current_file")
    
    # 1. 다른 파일 작업 중이었다면, 현재 작업 내용(모든 탭)을 먼저 자동 저장
    if old_filename and old_filename != new_filename:
        old_file_path = os.path.join(DATA_DIR, old_filename)
        try:
            with open(old_file_path, "r", encoding="utf-8") as f:
                old_raw_content = f.read()
            # preprocess_record가 3개의 값을 반환하므로 정확하게 받음
            _, old_source_text, _ = preprocess_record(old_raw_content)
            save_all_tabs_to_file(old_filename, old_source_text)
        except Exception as e:
            st.warning(f"이전 파일 '{old_filename}' 자동 저장 실패: {e}")

    # 2. 파일 전환에 필요한 상태(선택된 그룹)는 남기고, 나머지만 초기화
    persistent_group = st.session_state.get("selected_group", "All")
    
    # ✨✨✨ 핵심 수정: 데이터 상태를 먼저 명시적으로 초기화 (tab2_expanded_annotations 추가)
    st.session_state["tab1_annotations"] = []
    st.session_state["tab2_expanded_annotations"] = [] # [요청사항 수정]
    st.session_state["tab2_refined_terms"] = []
    st.session_state["tab3_final_codes"] = {}
    st.session_state["page_num"] = 1
    
    st.session_state.clear() # 모든 상태 초기화
    st.session_state["selected_group"] = persistent_group # 그룹 선택 상태 복원
    st.session_state["current_file"] = new_filename      # 새 파일 이름 설정
    
    # 데이터 상태 다시 초기화 (clear 후 재설정)
    st.session_state["tab1_annotations"] = []
    st.session_state["tab2_expanded_annotations"] = [] # [요청사항 수정]
    st.session_state["tab2_refined_terms"] = []
    st.session_state["tab3_final_codes"] = {}
    st.session_state["page_num"] = 1
    
    # ✨✨✨ 추가: Tab 1, 2, 3의 모든 widget 키 명시적 제거
    keys_to_remove = [key for key in list(st.session_state.keys()) 
                      if (key.startswith('entity_') or 
                          key.startswith('label_') or 
                          key.startswith('matches_') or
                          key.startswith('save_') or
                          key.startswith('delete_') or
                          key.startswith('confirm_save_') or
                          key.startswith('select_') or
                          key.startswith('tab3_') or
                          key.startswith('data_editor_') or # [요청사항 수정]
                          key.startswith('tab2_') or           # [요청사항 수정]
                          key.startswith('rep_selector_') or
                          key.startswith('action_') or
                          key.startswith('confirm_') or
                          key.startswith('delete_group_') or
                          key.startswith('merge_') or
                          # ✨ 파일명 포함된 키도 제거
                          '_entity_' in key or
                          '_label_' in key or
                          '_matches_' in key or
                          '_save_' in key or
                          '_delete_' in key or
                          '_confirm_save_' in key or
                          '_select_' in key or
                          '_tab3_' in key)]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # 3. 새 파일에 대한 기존 작업 내용 불러오기
    output_files = glob.glob(os.path.join(OUTPUT_DIR, f"{os.path.splitext(new_filename)[0]}_*.json"))
    if output_files:
        try:
            latest_file = max(output_files, key=os.path.getctime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # ✨✨✨ 명시적으로 데이터 설정
                loaded_annotations = data.get("tab1_annotations", [])
                
                # [요청사항 수정] 
                # 1. tab2_expanded_annotations 로드 시도
                # 2. 이 키가 없는 구버전 JSON이면 빈 리스트로 초기화
                loaded_expanded_annotations = data.get("tab2_expanded_annotations") 
                loaded_refined_terms = data.get("tab2_refined_terms", [])
                loaded_final_codes = data.get("tab3_final_codes", {})
                
                st.session_state["tab1_annotations"] = loaded_annotations if loaded_annotations else []
                
                # [요청사항 수정] 
                # 1. 'tab2_expanded_annotations'가 None이 아니면 (키가 존재하면) 로드
                # 2. None이면 (키가 없는 구버전) 빈 리스트로 설정
                #    -> 이 경우 loaded_refined_terms가 3단계에 바로 표시됨 (호환성)
                if loaded_expanded_annotations is not None:
                    st.session_state["tab2_expanded_annotations"] = loaded_expanded_annotations
                else:
                    st.session_state["tab2_expanded_annotations"] = [] # 구버전 파일
                
                st.session_state["tab2_refined_terms"] = loaded_refined_terms if loaded_refined_terms else []
                st.session_state["tab3_final_codes"] = loaded_final_codes if loaded_final_codes else {}
                
        except Exception as e:
            st.warning(f"파일 '{new_filename}' 불러오기 실패: {e}")
            # 불러오기 실패 시 명시적 초기화
            st.session_state["tab1_annotations"] = []
            st.session_state["tab2_expanded_annotations"] = [] # [요청사항 수정]
            st.session_state["tab2_refined_terms"] = []
            st.session_state["tab3_final_codes"] = {}
    else:
        # ✨✨✨ 추가: JSON이 없는 경우 명확히 초기화
        st.session_state["tab1_annotations"] = []
        st.session_state["tab2_expanded_annotations"] = [] # [요청사항 수정]
        st.session_state["tab2_refined_terms"] = []
        st.session_state["tab3_final_codes"] = {}
            
    # 4. 로드 후에도 필요한 state가 없으면 기본값으로 초기화
    if not st.session_state.get("tab1_annotations"):
        st.session_state["tab1_annotations"] = [{"entity": "", "label": "Symptom", "start": -1, "end": -1, "id": 0}]
    if "tab2_expanded_annotations" not in st.session_state: # [요청사항 수정]
        st.session_state.tab2_expanded_annotations = []
    if "tab2_refined_terms" not in st.session_state: 
        st.session_state.tab2_refined_terms = []
    if "tab3_final_codes" not in st.session_state: 
        st.session_state.tab3_final_codes = {}
    if "page_num" not in st.session_state:
        st.session_state['page_num'] = 1

# --- st.session_state 초기화 ---
# [요청사항 수정] tab2_expanded_annotations 추가
if "current_file" not in st.session_state:
    st.session_state.update({"current_file": None, "selected_group": "All", "tab1_annotations": [], "tab2_expanded_annotations": [], "tab2_refined_terms": [], "tab3_final_codes": {}})

# --- 폴더 생성 ---
for dir_path in [DATA_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path): os.makedirs(dir_path)

with st.sidebar:
    st.header("🗂️ 파일 선택")
    
    # ✨ 추가: 앱 시작 시 progress.json 정리
    if 'progress_cleaned' not in st.session_state:
        removed_count = clean_progress_data()
        st.session_state.progress_cleaned = True
        if removed_count > 0:
            st.toast(f"🧹 {removed_count}개의 삭제된 파일 정보를 정리했습니다.")
    
    progress_data = load_progress()
    try: all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    except FileNotFoundError: all_files = []
    
    # ✨ 수정: 앞의 숫자를 추출하는 함수
    def get_prefix_number(filename):
        """파일명 앞의 숫자를 추출 (예: '1_Long_22.txt' → 1)"""
        match = re.search(r'^(\d+)_', filename)
        if match:
            return int(match.group(1))
        return 999999  # 숫자가 없으면 뒤로 보냄
    
    # ✨ 수정: 그룹별로 분류하고 각 그룹 내에서 앞 숫자로 정렬
    file_groups = { 
        "Very-Long": sorted([f for f in all_files if 'Very-Long' in f], key=get_prefix_number), 
        "Long": sorted([f for f in all_files if re.search(r'^(\d+_)?Long', f) and 'Very-Long' not in f], key=get_prefix_number), 
        "Medium": sorted([f for f in all_files if 'Medium' in f], key=get_prefix_number), 
        "Short": sorted([f for f in all_files if 'Short' in f], key=get_prefix_number),
    }
    
    group_options = ["All"] + [name for name, files in file_groups.items() if files]
    selected_group = sac.segmented(items=group_options, index=group_options.index(st.session_state.get("selected_group", "All")), return_index=False, key='group_selector')
    if selected_group != st.session_state.get("selected_group"):
        st.session_state["selected_group"] = selected_group; st.session_state["current_file"] = None; st.rerun()
    
    # ✨ 수정: 파일 정렬 로직
    if selected_group == "All":
        # All 선택 시: 그룹 순서 유지, 각 그룹 내에서는 앞 숫자 오름차순
        display_files = []
        for group_name in ["Very-Long", "Long", "Medium", "Short"]:
            if group_name in file_groups and file_groups[group_name]:
                display_files.extend(file_groups[group_name])
    else:
        # 특정 그룹 선택 시: 앞 숫자 오름차순
        display_files = file_groups.get(selected_group, [])
    
    def format_file_name(filename):
        is_completed = "✅" if filename in progress_data and progress_data[filename] == 'completed' else "⏳"
        base_name = os.path.splitext(filename)[0]
        return f"{is_completed} {base_name}"
    current_index = None
    if st.session_state.get("current_file") and st.session_state.get("current_file") in display_files:
        current_index = display_files.index(st.session_state.get("current_file"))
    selected_file_from_box = st.selectbox("작업할 파일을 검색하거나 선택하세요.", options=display_files, format_func=format_file_name, index=current_index, placeholder="파일 선택...")
    if selected_file_from_box and selected_file_from_box != st.session_state.get("current_file"):
        handle_file_select(selected_file_from_box); st.rerun()
    if current_index is not None:
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬅️ 이전 파일", use_container_width=True, disabled=(current_index <= 0)):
                handle_file_select(display_files[current_index - 1]); st.rerun()
        with nav_col2:
            if st.button("다음 파일 ➡️", use_container_width=True, disabled=(current_index >= len(display_files) - 1)):
                handle_file_select(display_files[current_index + 1]); st.rerun()
    st.markdown("---")
    st.markdown("<h2 style='font-size: 1.1rem;'>📊 작업 진행 상황</h2>", unsafe_allow_html=True)
    
    # ✨ 수정: 실제 존재하는 파일만 카운트
    total_files = len(all_files)
    completed_files = len([f for f in all_files if f in progress_data and progress_data[f] == 'completed'])
    
    st.write(f"**전체 진행률 ({completed_files}/{total_files})**")
    st.progress(completed_files / total_files if total_files > 0 else 0)
    
    for group_name, files_in_group in file_groups.items():
        if files_in_group:
            # ✨ 수정: 그룹별로도 실제 존재하는 파일만 카운트
            completed_in_group = len([f for f in files_in_group if f in progress_data and progress_data[f] == 'completed'])
            total_in_group = len(files_in_group)
            st.write(f"{group_name} ({completed_in_group}/{total_in_group})")
            st.progress(completed_in_group / total_in_group if total_in_group > 0 else 0)

# =====================================================================
# --- 메인 페이지 (Main Page) UI 구성 ---
# =====================================================================
DISPLAY_TEXT, SOURCE_TEXT, SOURCE_TO_DISPLAY_MAP = "", "", []
if st.session_state.get("current_file"):
    file_path = os.path.join(DATA_DIR, st.session_state["current_file"])
    try:
        with open(file_path, "r", encoding="utf-8") as f: RAW_CONTENT = f.read()
        DISPLAY_TEXT, SOURCE_TEXT, SOURCE_TO_DISPLAY_MAP = preprocess_record(RAW_CONTENT)
    except Exception as e:
        st.error(f"파일 처리 오류: {e}")

if DISPLAY_TEXT:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.header(f"기록지: `{st.session_state['current_file']}`")
        annotated_html = get_annotated_html(DISPLAY_TEXT, SOURCE_TO_DISPLAY_MAP, st.session_state.get("tab1_annotations", []))
        st.components.v1.html(annotated_html, height=800, scrolling=True)

    with col2:
        tab1, tab2, tab3 = st.tabs(["**1) Annotation**", "**2) Refining**", "**3) Coding**"])
        with tab1:
            # (탭 1 코드는 기존과 동일)
            st.info("기록지 내에서 코딩 대상이 되는 진단명과 증상을 라벨링합니다.\n- 동일한 개념이라도 모두 라벨링합니다.")
            if 'page_num' not in st.session_state: st.session_state.page_num = 1
            ITEMS_PER_PAGE = 4
            annotations = st.session_state.get("tab1_annotations", [])
            total_items = len(annotations)
            total_pages = math.ceil(total_items / ITEMS_PER_PAGE) if total_items > 0 else 1
            start_idx = (st.session_state.page_num - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            grid_annotations = [annotations[i:i + 2] for i in range(start_idx, min(end_idx, total_items), 2)]
            for row_items in grid_annotations:
                row_cols = st.columns(2)
                for j, annotation in enumerate(row_items):
                    try: actual_index = annotations.index(annotation)
                    except ValueError: continue
                    with row_cols[j]:
                        st.markdown(f"**Annotation #{actual_index + 1}**")
                        # ✨✨✨ 수정: widget 키에 파일명 포함
                        current_file = st.session_state.get("current_file", "default")
                        entity_text = st.text_input("Entity", value=annotation.get("entity", ""), key=f"{current_file}_entity_{actual_index}")
                        label_items = ["Symptom", "Diagnosis", "Supportive"]
                        try: default_index = label_items.index(annotation.get("label", "Symptom"))
                        except ValueError: default_index = 0
                        label_type = sac.segmented(items=label_items, label="Label Type", key=f"{current_file}_label_{actual_index}", index=default_index, return_index=False)
                        if f"{current_file}_matches_{actual_index}" not in st.session_state: st.session_state[f"{current_file}_matches_{actual_index}"] = []
                        if len(st.session_state.get(f"{current_file}_matches_{actual_index}", [])) > 1:
                            match_options = [f"위치 {k+1} (인덱스: {m['start']}): ...{SOURCE_TEXT[max(0, m['start']-15):m['start']]}**{SOURCE_TEXT[m['start']:m['end']]}**{SOURCE_TEXT[m['end']:min(len(SOURCE_TEXT), m['end']+15)]}..." for k, m in enumerate(st.session_state[f"{current_file}_matches_{actual_index}"])]
                            selected_option = st.radio("...", options=match_options, key=f"{current_file}_select_{actual_index}", label_visibility="collapsed")
                            if st.button("📍 위치 확정 및 저장", key=f"{current_file}_confirm_save_{actual_index}", use_container_width=True):
                                selected_start_index = int(re.search(r"인덱스: (\d+)", selected_option).group(1))
                                st.session_state.tab1_annotations[actual_index] = {"entity": entity_text, "label": label_type, "start": selected_start_index, "end": selected_start_index + len(entity_text), "id": annotation.get('id', actual_index)}
                                st.session_state[f"{current_file}_matches_{actual_index}"] = []
                                save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                                # ✨✨✨ 수정: 고유 ID 생성 함수 사용
                                if actual_index == len(st.session_state.tab1_annotations) - 1: 
                                    next_id = get_next_annotation_id()
                                    st.session_state.tab1_annotations.append({"entity": "", "label": "Symptom", "start": -1, "end": -1, "id": next_id});
                                st.rerun()
                        else:
                            if st.button("💾 임시 저장", key=f"{current_file}_save_{actual_index}", use_container_width=True):
                                if entity_text:
                                    search_pattern = r'\s+'.join(re.escape(part) for part in entity_text.split())
                                    matches = [{"start": m.start(), "end": m.end()} for m in re.finditer(search_pattern, SOURCE_TEXT)]
                                    if len(matches) == 0: st.error(f"'{entity_text}'를 찾을 수 없습니다.")
                                    elif len(matches) == 1:
                                        st.session_state.tab1_annotations[actual_index] = {"entity": entity_text, "label": label_type, "start": matches[0]["start"], "end": matches[0]["end"], "id": annotation.get('id', actual_index)}
                                        save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                                        # ✨✨✨ 수정: 고유 ID 생성 함수 사용
                                        if actual_index == len(st.session_state.tab1_annotations) - 1: 
                                            next_id = get_next_annotation_id()
                                            st.session_state.tab1_annotations.append({"entity": "", "label": "Symptom", "start": -1, "end": -1, "id": next_id});
                                        st.rerun()
                                    else: st.session_state[f"{current_file}_matches_{actual_index}"] = matches; st.rerun()
                                else: st.warning("Entity를 입력해주세요.")
                            if st.button("🗑️ 삭제", key=f"{current_file}_delete_{actual_index}", use_container_width=True):
                                st.session_state.tab1_annotations.pop(actual_index)
                                if f"{current_file}_matches_{actual_index}" in st.session_state: del st.session_state[f"{current_file}_matches_{actual_index}"]
                                # ✨✨✨ 수정: 빈 리스트가 되면 고유 ID로 새 항목 추가
                                if len(st.session_state.tab1_annotations) == 0: 
                                    st.session_state.tab1_annotations.append({"entity": "", "label": "Symptom", "start": -1, "end": -1, "id": 0})
                                save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT); st.rerun()
                        st.caption(f"Saved (Source Index) - Start: {annotation.get('start', -1)}, End: {annotation.get('end', -1)}")
            st.markdown("---")
            total_pages = math.ceil(len(annotations) / ITEMS_PER_PAGE) if annotations else 1
            page_col1, page_col2, page_col3 = st.columns([1, 1, 1])
            with page_col1:
                if st.button("⬅️ 이전", key='prev_page', use_container_width=True):
                    if st.session_state.page_num > 1: st.session_state.page_num -= 1; st.rerun()
            with page_col3:
                if st.button("다음 ➡️", key='next_page', use_container_width=True):
                    if st.session_state.page_num < total_pages: st.session_state.page_num += 1; st.rerun()
            if total_pages > 0: st.write(f"페이지: {st.session_state.page_num} / {total_pages}", )

        # =====================================================================
        # --- [요청사항 수정] Tab 2 로직 전면 수정 ---
        # =====================================================================
        with tab2:
            st.header("Refining (용어 정제 및 그룹화)")
            st.info("1단계(약어 확장) -> 2단계(그룹화) -> 3단계(최종) 순서로 진행됩니다.")

            # --- Tab 2 상태 초기화 ---
            if 'tab2_expanded_annotations' not in st.session_state: 
                st.session_state.tab2_expanded_annotations = []
            if 'tab2_refined_terms' not in st.session_state: 
                st.session_state.tab2_refined_terms = []

            # --- 데이터 준비 ---
            # 1. Tab 1에서 유효한 모든 annotation
            valid_annotations = [anno for anno in st.session_state.get("tab1_annotations", []) if anno.get("entity")]
            
            # 2. 이미 1단계(확장)를 통과한 항목들의 ID
            all_expanded_ids = {item['id'] for item in st.session_state.tab2_expanded_annotations}
            
            # 3. 이미 2단계(그룹화)를 통과한 항목들의 ID
            all_refined_ids = {
                item['id'] 
                for group in st.session_state.tab2_refined_terms 
                for item in group['original_entities']
            }

            # --- 1단계 UI: 약어 확장 (미처리 Annotation) ---
            st.markdown("---")
            st.subheader("1단계: 약어 확장 (미처리 Annotation)")
            
            # 1단계 대상: (Tab 1 유효 항목) - (1단계 통과 항목) - (2단계 통과 항목)
            unprocessed_annotations = [
                anno for anno in valid_annotations 
                if anno.get('id') not in all_expanded_ids and anno.get('id') not in all_refined_ids
            ]

            if not unprocessed_annotations:
                st.success("1단계: 처리할 원본 Annotation이 없습니다.")
            else:
                df_data_s1 = [{"select": False, "id": anno.get('id'), "entity": anno['entity'], "label": anno['label']} for anno in unprocessed_annotations]
                edited_df_s1 = st.data_editor(
                    df_data_s1, 
                    key="data_editor_s1_annotations", 
                    column_config={
                        "select": st.column_config.CheckboxColumn(default=False),
                        "id": None 
                    }, 
                    hide_index=True, 
                    use_container_width=True
                )
                
                selected_rows_s1 = [row for row in edited_df_s1 if row['select']]
                
                # --- 1단계 액션 패널 ---
                if selected_rows_s1:
                    # '그대로 확정' 버튼 (여러 개 선택 가능)
                    if st.button(f"✅ 선택한 {len(selected_rows_s1)}개 항목 '그대로 확정' (확장 없음)", type="secondary"):
                        selected_ids_s1 = {row['id'] for row in selected_rows_s1}
                        for anno in unprocessed_annotations:
                            if anno.get('id') in selected_ids_s1:
                                # 원본 annotation을 복사하고 expanded_term 추가
                                expanded_anno = anno.copy()
                                expanded_anno['expanded_term'] = anno['entity'] # 그대로 확정
                                st.session_state.tab2_expanded_annotations.append(expanded_anno)
                        save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                        st.rerun()

                    # '약어 펼치기' (1개 선택 시)
                    if len(selected_rows_s1) == 1:
                        st.markdown("**1개 항목 약어 펼치기:**")
                        selected_entity = selected_rows_s1[0]['entity']
                        
                        # 자동 약어 매핑 (예: 'Rt.' -> 'Right')
                        # 간단한 공백/특수문자 기반 분리 후 매핑
                        words = re.split(r'(\s+|[(),])', selected_entity)
                        expanded_words = []
                        for word in words:
                            if not word: continue
                            clean_word = word.lower()
                            # 'rt.' -> 'Right'
                            mapped = ABBREVIATION_MAPPING.get(clean_word, None)
                            if mapped:
                                # 대소문자 유지 (첫 글자만 대문자)
                                expanded_words.append(mapped[0].upper() + mapped[1:] if mapped else word)
                            # 'tenderness(+)' -> 'tenderness'
                            elif clean_word.endswith('(+)'):
                                expanded_words.append(word[:-3])
                            elif clean_word.endswith('(-)'):
                                expanded_words.append(word[:-3])
                            else:
                                expanded_words.append(word)
                        
                        # 공백 재조합
                        default_full_term = "".join(expanded_words)
                        # 'abdominal pain' 같은 일반 용어 매핑도 확인
                        if default_full_term.lower() in ABBREVIATION_MAPPING:
                             default_full_term = ABBREVIATION_MAPPING[default_full_term.lower()]

                        rep_term = st.text_input(
                            "Full-Term 입력:", 
                            value=default_full_term,
                            placeholder=f"{selected_entity}의 Full-Term",
                            key="tab2_s1_full_term_input"
                        )
                        
                        if st.button("✔️ 확장 확정", key="confirm_s1_expand", disabled=not rep_term):
                            selected_id_s1 = selected_rows_s1[0]['id']
                            original_anno = next(anno for anno in unprocessed_annotations if anno.get('id') == selected_id_s1)
                            
                            expanded_anno = original_anno.copy()
                            expanded_anno['expanded_term'] = rep_term.strip() # 확장된 용어 저장
                            st.session_state.tab2_expanded_annotations.append(expanded_anno)
                            
                            save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                            st.rerun()
                    
                    elif len(selected_rows_s1) > 1:
                        st.info("약어 펼치기는 한 항목씩만 가능합니다.")
                
                else:
                    st.write("테이블에서 작업을 원하는 항목을 선택하세요.")

            # --- 2단계 UI: 용어 그룹화 (확장 완료) ---
            st.markdown("---")
            st.subheader("2단계: 용어 그룹화 (확장 완료)")

            # 2단계 대상: (1단계 통과 항목) - (2단계 통과 항목)
            annotations_to_group = [
                anno for anno in st.session_state.tab2_expanded_annotations 
                if anno.get('id') not in all_refined_ids
            ]

            if not annotations_to_group:
                st.success("2단계: 그룹화할 항목이 없습니다.")
            else:
                # [요청사항 수정] expanded_term을 기본으로 보여줌
                df_data_s2 = [
                    {
                        "select": False, 
                        "id": anno.get('id'), 
                        "expanded_term": anno.get('expanded_term', anno.get('entity')), # expanded_term 우선
                        "label": anno['label'],
                        "original_entity": anno['entity'] # 원문 참고용
                    } 
                    for anno in annotations_to_group
                ]
                edited_df_s2 = st.data_editor(
                    df_data_s2, 
                    key="data_editor_s2_grouping", 
                    column_config={
                        "select": st.column_config.CheckboxColumn("선택", default=False),
                        "id": None,
                        "expanded_term": st.column_config.TextColumn("정제된 용어 (그룹화 대상)"),
                        "original_entity": st.column_config.TextColumn("원본"),
                    }, 
                    hide_index=True, 
                    use_container_width=True
                )
                
                selected_rows_s2 = [row for row in edited_df_s2 if row['select']]

                # --- 2단계 액션 패널 ---
                if selected_rows_s2:
                    # '단독 그룹 확정' (1개 이상 선택 가능)
                    if st.button(f"➡️ 선택한 {len(selected_rows_s2)}개 항목 '단독 그룹'으로 확정", type="secondary"):
                        selected_ids_s2 = {row['id'] for row in selected_rows_s2}
                        items_to_move = []
                        remaining_items = []
                        
                        for anno in st.session_state.tab2_expanded_annotations:
                            if anno.get('id') in selected_ids_s2:
                                items_to_move.append(anno)
                            else:
                                remaining_items.append(anno)
                                
                        # 각 항목을 단독 그룹으로 만들어 3단계로 보냄
                        for item in items_to_move:
                            st.session_state.tab2_refined_terms.append({
                                "term": item['expanded_term'], # 대표 용어 = 정제된 용어
                                "original_entities": [item]    # 본인만 포함
                            })
                        
                        st.session_state.tab2_expanded_annotations = remaining_items # 1단계 목록에서 제거
                        save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                        st.rerun()

                    # '그룹으로 묶기' (2개 이상 선택 시)
                    if len(selected_rows_s2) > 1:
                        st.markdown(f"**{len(selected_rows_s2)}개 항목 그룹화:**")
                        
                        # [요청사항 수정] expanded_term을 기준으로 대표 지정
                        df_selected_data_s2 = [
                            {
                                "is_rep": False, 
                                "expanded_term": row['expanded_term'], 
                                "label": row['label'],
                                "original_entity": row['original_entity']
                            } 
                            for row in selected_rows_s2
                        ]
                        edited_rep_df = st.data_editor(
                            df_selected_data_s2, 
                            key="rep_selector_df_s2", 
                            column_config={
                                "is_rep": st.column_config.CheckboxColumn("대표 지정"),
                                "expanded_term": st.column_config.TextColumn("정제된 용어"),
                                "original_entity": st.column_config.TextColumn("원본"),
                            }, 
                            hide_index=True
                        )
                        
                        rep_term_from_radio = [row['expanded_term'] for row in edited_rep_df if row['is_rep']]
                        new_rep_term = st.text_input("새 대표 용어 직접 입력 (선택 사항):", key="tab2_s2_rep_term_input")
                        final_rep_term = new_rep_term if new_rep_term else (rep_term_from_radio[0] if len(rep_term_from_radio) == 1 else None)
                        
                        if st.button("➕ 선택 항목을 그룹으로 묶기", key="confirm_s2_group", disabled=not final_rep_term):
                            selected_ids_s2 = {row['id'] for row in selected_rows_s2}
                            original_entities_to_add = []
                            remaining_items = []
                            
                            # 1단계 목록에서 선택된 항목들 분리
                            for anno in st.session_state.tab2_expanded_annotations:
                                if anno.get('id') in selected_ids_s2:
                                    original_entities_to_add.append(anno)
                                else:
                                    remaining_items.append(anno)

                            # 3단계로 보낼 새 그룹 생성
                            st.session_state.tab2_refined_terms.append({
                                "term": final_rep_term, 
                                "original_entities": original_entities_to_add
                            })
                            
                            st.session_state.tab2_expanded_annotations = remaining_items # 1단계 목록에서 제거
                            save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                            st.rerun()
                    
                    elif len(selected_rows_s2) == 1:
                        st.info("그룹으로 묶으려면 2개 이상 항목을 선택하세요.")
                
                else:
                    st.write("테이블에서 작업을 원하는 항목을 선택하세요.")


            # --- 3단계 UI: 정제된 용어 그룹 (최종) ---
            st.markdown("---")
            st.subheader("3단계: 정제된 용어 그룹 (최종)")
            
            if not st.session_state.get('tab2_refined_terms', []): 
                st.write("아직 생성된 그룹이 없습니다.")
                
            # [요청사항 수정] 거꾸로 순회해야 pop 에러 없음
            for i in reversed(range(len(st.session_state.get('tab2_refined_terms', [])))):
                group = st.session_state.tab2_refined_terms[i]
                
                with st.expander(f"**대표 용어:** {group['term']}  (포함된 용어: {len(group['original_entities'])})"):
                    st.markdown("**포함된 용어:**")
                    for item in group['original_entities']:
                        # [요청사항 수정] expanded_term과 entity(원문) 동시 표시
                        expanded = item.get('expanded_term', 'N/A')
                        original = item.get('entity', 'N/A')
                        
                        if expanded == original:
                            st.markdown(f"- {expanded} `({item.get('label', 'N/A')})`")
                        else:
                            st.markdown(f"- {expanded} (원문: {original}) `({item.get('label', 'N/A')})`")
                    
                    st.markdown("---")
                    st.warning("그룹 삭제 시, 포함된 항목들은 재처리(1단계 또는 2단계)를 위해 이동됩니다.")
                    if st.button("그룹 삭제 (재처리)", key=f"delete_group_{i}", type="secondary"):
                        
                        # [요청사항 수정] 재처리 로직
                        popped_group = st.session_state.tab2_refined_terms.pop(i)
                        
                        for item in popped_group['original_entities']:
                            # expanded_term 키가 있으면 (새 워크플로우) -> 2단계로 이동
                            if 'expanded_term' in item:
                                st.session_state.tab2_expanded_annotations.append(item)
                            # expanded_term 키가 없으면 (구버전 데이터) -> 1단계로 이동
                            # (아무데도 추가하지 않으면, 1단계 필터 로직이 자동으로 감지함)
                            else:
                                pass # 1단계로 자동 이동됨
                        
                        save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                        st.rerun()

        with tab3:
            st.header("Coding (KCD 코드 부여)")
            st.info("각 용어에 KCD 코드와 설명을 입력하고 하단의 '전체 저장' 버튼을 클릭하세요.")
            
            # 초기화
            if 'tab3_final_codes' not in st.session_state or not isinstance(st.session_state.tab3_final_codes, dict):
                st.session_state.tab3_final_codes = {"before_merge": [], "after_merge": []}
            
            # [요청사항 수정] Tab 2의 3단계(최종 그룹)의 'term'을 사용
            refined_terms = [group['term'] for group in st.session_state.get('tab2_refined_terms', [])]
            
            # ==================== 1단계: 입력 폼 & 저장된 데이터 테이블 ====================
            st.subheader("1단계: 코드 입력")
            
            if not refined_terms:
                st.warning("Tab2의 3단계(최종 그룹)가 없습니다. Tab2를 먼저 완료해주세요.")
            else:
                # 현재 저장된 데이터 (Term을 키로 사용)
                saved_data_dict = {item['Term']: item for item in st.session_state.tab3_final_codes.get('before_merge', [])}
                
                st.markdown("##### 📝 코드 및 코드설명 작성")
                st.markdown("아래에서 모든 용어에 대한 코드를 입력하세요.")
                
                current_file = st.session_state.get("current_file", "default")
                
                # 모든 Term에 대한 입력 폼 (저장 버튼 없이)
                for i, term in enumerate(refined_terms):
                    # 기존 저장된 데이터 확인
                    saved_item = saved_data_dict.get(term, {})
                    term_lower = term.lower()
                    
                    # 기본값 설정: 1) 저장된 데이터 2) 매핑 3) 빈 값
                    if saved_item and saved_item.get('Term') == term:
                        default_select = saved_item.get('select', False)
                        default_code = saved_item.get('KCD Code', '')
                        default_desc = saved_item.get('KCD Description', '')
                    elif term_lower in TERM_TO_KCD_MAPPING:
                        default_select = False
                        default_code = TERM_TO_KCD_MAPPING[term_lower]['code']
                        default_desc = TERM_TO_KCD_MAPPING[term_lower]['desc']
                    else:
                        default_select = False
                        default_code = ''
                        default_desc = ''
                    
                    # 구분선
                    if i > 0:
                        st.markdown("---")
                    
                    st.markdown(f"**Term: {term}**")
                    
                    col1, col2, col3 = st.columns([1, 2, 3])
                    
                    with col1:
                        st.checkbox("병합 대상", value=default_select, key=f"{current_file}_tab3_form_select_{term}")
                    
                    with col2:
                        st.text_input("KCD Code", value=default_code, key=f"{current_file}_tab3_form_code_{term}", 
                                     placeholder="예: R50.99", label_visibility="collapsed")
                    
                    with col3:
                        st.text_input("KCD Description", value=default_desc, key=f"{current_file}_tab3_form_desc_{term}", 
                                     placeholder="예: Fever, unspecified", label_visibility="collapsed")
                
                st.markdown("---")
                
                # ✨✨✨ 전체 저장 버튼 (하나만)
                if st.button("💾 저장", type="primary", use_container_width=True):
                    # 모든 Term의 입력값 수집
                    new_before_merge = []
                    
                    for i, term in enumerate(refined_terms):
                        merge_select = st.session_state.get(f"{current_file}_tab3_form_select_{term}", False)
                        code_input = st.session_state.get(f"{current_file}_tab3_form_code_{term}", "")
                        desc_input = st.session_state.get(f"{current_file}_tab3_form_desc_{term}", "")
                        
                        # 값이 있는 것만 저장 (빈 값은 제외)
                        if code_input.strip() or desc_input.strip():
                            new_before_merge.append({
                                "select": merge_select,
                                "Term": term,
                                "KCD Code": code_input.strip(),
                                "KCD Description": desc_input.strip()
                            })
                    
                    # before_merge 업데이트
                    st.session_state.tab3_final_codes['before_merge'] = new_before_merge
                    
                    # 파일 저장
                    save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                    st.success(f"✅ {len(new_before_merge)}개 항목이 저장되었습니다!")
                    st.rerun()
                
                st.markdown("---")
                
                # 저장된 데이터 테이블 (읽기 전용)
                st.markdown("##### 📋 저장된 코드 목록")
                
                before_merge_data = st.session_state.tab3_final_codes.get('before_merge', [])
                if not before_merge_data:
                    st.info("아직 저장된 코드가 없습니다. 위에서 입력 후 '전체 저장' 버튼을 클릭하세요.")
                else:
                    # DataFrame으로 표시
                    display_data = []
                    for item in before_merge_data:
                        display_data.append({
                            "병합 대상": "☑" if item.get('select', False) else "☐",
                            "Term": item.get('Term', ''),
                            "KCD Code": item.get('KCD Code', ''),
                            "KCD Description": item.get('KCD Description', '')
                        })
                    
                    df_before = pd.DataFrame(display_data)
                    st.dataframe(df_before, use_container_width=True, hide_index=True)
                    
                    st.caption("💡 수정하려면 위의 입력 폼에서 다시 입력하고 '전체 저장'을 클릭하세요.")
                
                st.markdown("---")
                
                # ==================== 2단계: 코드 수정 및 병합 ====================
                st.subheader("2단계: 코드 병합")
                
                # ✨✨✨ 수정: before_merge에서 데이터 가져오기 (widget이 아닌 저장된 데이터에서)
                current_data = st.session_state.tab3_final_codes.get('before_merge', [])
                
                if not current_data:
                    st.info("1단계에서 코드를 먼저 저장해주세요.")
                else:
                    # 선택된 항목 확인
                    selected_items = [item for item in current_data if item.get('select', False)]
                    
                    if not selected_items:
                        # 선택 안 함: 확정 버튼만 표시
                        st.info("병합할 항목이 없으면 바로 확정할 수 있습니다.")
                        if st.button("➡️ 수정 없이 최종 코드로 확정", use_container_width=True, type="primary"):
                            st.session_state.tab3_final_codes['after_merge'] = [
                                {
                                    "Term": item['Term'],
                                    "KCD Code": item['KCD Code'],
                                    "KCD Description": item['KCD Description'],
                                    "is_merged": False
                                }
                                for item in current_data
                            ]
                            save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                            st.success("최종 코드로 확정되었습니다!")
                            st.rerun()
                    else:
                        # 선택됨: 병합 UI 표시
                        st.info(f"**{len(selected_items)}개 항목 선택됨**: 병합 그룹을 만들어 새로운 코드로 통합할 수 있습니다.")
                        
                        st.markdown("---")
                        
                        # 병합 그룹 관리
                        if 'merge_groups' not in st.session_state:
                            st.session_state.merge_groups = []
                        
                        # 1단계에서 선택된 코드 조합 확인
                        selected_codes = frozenset([
                            item['KCD Code'] 
                            for item in current_data 
                            if item.get('select', False) and item.get('KCD Code', '').strip()
                        ])
                        
                        # 매핑 확인
                        auto_mapped = selected_codes in MERGE_CODE_MAPPING
                        if auto_mapped:
                            mapped_data = MERGE_CODE_MAPPING[selected_codes]
                            default_term = mapped_data['term']
                            default_code = mapped_data['code']
                            default_desc = mapped_data['desc']
                            st.success(f"💡 자동 매핑됨: {default_term} ({default_code})")
                        else:
                            default_term = ''
                            default_code = ''
                            default_desc = ''
                        
                        # 병합 그룹 생성 UI
                        st.write("**병합 그룹 생성**")
                        
                        # 자동 매핑된 경우 간소화된 UI
                        if auto_mapped:
                            st.write("자동 매핑된 병합:")
                            st.markdown(f"- **새 Term:** {default_term}")
                            st.markdown(f"- **새 Code:** {default_code}")
                            st.markdown(f"- **새 Desc:** {default_desc}")
                            st.markdown(f"- **포함 항목:** {', '.join([item['Term'] for item in selected_items])}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("✅ 이대로 그룹 추가", use_container_width=True, type="primary"):
                                    # 선택된 모든 항목의 Term 추가
                                    selected_terms = [item['Term'] for item in selected_items]
                                    st.session_state.merge_groups.append({
                                        "terms": selected_terms,
                                        "new_term": default_term,
                                        "new_code": default_code,
                                        "new_desc": default_desc
                                    })
                                    st.success(f"병합 그룹 추가됨: {default_term}")
                                    st.rerun()
                            
                            with col2:
                                if st.button("✏️ 수정하기", key="edit_auto_mapped"):
                                    st.session_state['show_manual_form'] = True
                                    st.rerun()
                        
                        # 수동 입력 또는 자동 매핑 수정 모드
                        if not auto_mapped or st.session_state.get('show_manual_form', False):
                            with st.form("merge_group_form"):
                                st.write("새로운 병합 그룹 만들기:")
                                
                                # 선택된 항목들 중에서 그룹에 포함할 항목 선택
                                st.write("그룹에 포함할 항목 선택:")
                                group_selections = {}
                                for idx, item in enumerate(selected_items):
                                    # 자동 매핑된 경우 기본적으로 모두 체크
                                    default_checked = auto_mapped
                                    group_selections[item['Term']] = st.checkbox(
                                        f"{item['Term']} ({item['KCD Code']})", 
                                        value=default_checked,
                                        key=f"merge_form_select_{idx}"
                                    )
                                
                                # 새 코드 정보 입력
                                new_term = st.text_input("새로운 대표 용어", value=default_term, placeholder="예: Nausea with Vomiting")
                                new_code = st.text_input("새로운 KCD Code", value=default_code, placeholder="예: R11.3")
                                new_desc = st.text_input("새로운 KCD Description", value=default_desc, placeholder="예: Nausea with Vomiting")
                                
                                submitted = st.form_submit_button("➕ 이 그룹 추가")
                                
                                if submitted:
                                    selected_for_group = [term for term, is_selected in group_selections.items() if is_selected]
                                    if selected_for_group and new_term and new_code:
                                        st.session_state.merge_groups.append({
                                            "terms": selected_for_group,
                                            "new_term": new_term,
                                            "new_code": new_code,
                                            "new_desc": new_desc
                                        })
                                        # 수정 모드 해제
                                        if 'show_manual_form' in st.session_state:
                                            del st.session_state['show_manual_form']
                                        st.success(f"병합 그룹 추가됨: {new_term}")
                                        st.rerun()
                                    else:
                                        st.error("그룹에 포함할 항목과 새 코드 정보를 모두 입력해주세요.")
                        
                        # 생성된 병합 그룹 표시
                        if st.session_state.merge_groups:
                            st.markdown("---")
                            st.write("**생성된 병합 그룹:**")
                            for idx, group in enumerate(st.session_state.merge_groups):
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    terms_str = ", ".join(group['terms'])
                                    st.markdown(f"✅ **{group['new_term']}** ({group['new_code']})  \n포함 항목: {terms_str}")
                                with col2:
                                    if st.button("삭제", key=f"delete_merge_group_{idx}", type="secondary"):
                                        st.session_state.merge_groups.pop(idx)
                                        st.rerun()
                        
                        st.markdown("---")
                        
                        # 모든 병합 적용
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 모든 병합 적용하기", use_container_width=True, type="primary"):
                                # 병합 로직
                                merged_terms = set()
                                for group in st.session_state.merge_groups:
                                    merged_terms.update(group['terms'])
                                
                                # after_merge 생성
                                after_merge_data = []
                                
                                # 병합된 그룹 추가
                                for group in st.session_state.merge_groups:
                                    after_merge_data.append({
                                        "Term": group['new_term'],
                                        "KCD Code": group['new_code'],
                                        "KCD Description": group['new_desc'],
                                        "is_merged": True
                                    })
                                
                                # 병합되지 않은 항목 추가
                                for item in current_data:
                                    if item['Term'] not in merged_terms:
                                        after_merge_data.append({
                                            "Term": item['Term'],
                                            "KCD Code": item['KCD Code'],
                                            "KCD Description": item['KCD Description'],
                                            "is_merged": False
                                        })
                                
                                st.session_state.tab3_final_codes['after_merge'] = after_merge_data
                                st.session_state.merge_groups = []  # 병합 그룹 초기화
                                save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                                st.success("병합이 적용되었습니다!")
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ 병합 취소하고 확정", use_container_width=True, type="secondary"):
                                st.session_state.tab3_final_codes['after_merge'] = [
                                    {
                                        "Term": item['Term'],
                                        "KCD Code": item['KCD Code'],
                                        "KCD Description": item['KCD Description'],
                                        "is_merged": False
                                    }
                                    for item in current_data
                                ]
                                st.session_state.merge_groups = []
                                save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                                st.success("병합 없이 확정되었습니다!")
                                st.rerun()
                
                st.markdown("---")
                
                # ==================== 3단계: 최종 코딩 결과 ====================
                st.subheader("3단계: 최종 코딩 결과")
                
                after_merge_data = st.session_state.tab3_final_codes.get('after_merge', [])
                if not after_merge_data:
                    st.info("2단계에서 확정 버튼을 눌러주세요.")
                else:
                    # ✨ 수정: 볼드체만 적용
                    df_after = pd.DataFrame(after_merge_data, dtype=object)
                    
                    # 스타일 함수: 볼드체만 적용 (배경색 제거)
                    def highlight_merged(row):
                        if row.get('is_merged') == True:  # ✨ 명시적으로 True와 비교
                            return ['font-weight: bold;'] * len(row)
                        return [''] * len(row)
                    
                    # 표시할 컬럼 (is_merged는 제외)
                    columns_to_show = ["Term", "KCD Code", "KCD Description"]
                    display_df = df_after[columns_to_show].copy()
                    
                    # ✨ 수정: is_merged 정보를 사용하기 위해 전체 df에서 스타일 적용 후 선택
                    # 각 행에 대해 is_merged 확인하여 스타일 적용
                    def style_row(row):
                        idx = row.name
                        if df_after.loc[idx, 'is_merged'] == True:
                            return ['font-weight: bold;'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = display_df.style.apply(style_row, axis=1)
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    st.caption("💡 볼드체로 표시된 항목은 병합된 항목입니다.")

        st.markdown("---")
        st.subheader("작업 상태 변경")
        progress_data = load_progress()
        is_completed = st.session_state.get("current_file") in progress_data and progress_data[st.session_state.current_file] == 'completed'
        if is_completed:
            if st.button("🔄 작업 완료 취소", use_container_width=True, type="secondary", key="cancel_completion_button"):
                del progress_data[st.session_state.current_file]
                save_progress(progress_data)
                st.success(f"`{st.session_state.current_file}` 작업을 '진행 중' 상태로 변경했습니다."); st.rerun()
        else:
            if st.button("⭐ 이 파일 작업 완료로 표시", use_container_width=True, type="primary", key="mark_complete_button"):
                save_path = save_all_tabs_to_file(st.session_state["current_file"], SOURCE_TEXT)
                progress_data[st.session_state.get("current_file")] = "completed"
                save_progress(progress_data)
                st.success(f"`{st.session_state['current_file']}` 작업을 완료로 표시했습니다. (저장 위치: {save_path})")
                st.balloons(); st.rerun()
else:
    st.info("👈 사이드바에서 작업할 파일을 선택해주세요.")