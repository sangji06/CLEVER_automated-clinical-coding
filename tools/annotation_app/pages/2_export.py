import streamlit as st
import os
import json
import glob
from datetime import datetime

OUTPUT_DIR = "output"
PROGRESS_FILE = "progress.json"

def load_progress():
    """진행 상황 파일을 읽어옴"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

st.set_page_config(layout="wide")
st.title("💾 모든 완료된 작업 Export")
st.markdown("---")

st.info("이 페이지에서는 '작업 완료'로 표시된 모든 파일의 라벨링 결과를 하나의 JSON 파일로 취합하여 다운로드할 수 있습니다.")

progress_data = load_progress()
completed_files = list(progress_data.keys())

if not completed_files:
    st.warning("아직 '작업 완료'로 표시된 파일이 없습니다.")
else:
    st.write(f"**총 {len(completed_files)}개의 완료된 파일이 있습니다.**")
    st.json(completed_files, expanded=False)

    st.markdown("---")
    
    if st.button("📥 모든 완료된 결과 통합 다운로드", use_container_width=True):
        all_completed_annotations = []
        
        for filename in completed_files:
            # 각 파일에 해당하는 가장 최신 output json 찾기
            output_files = glob.glob(os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_*.json"))
            if output_files:
                latest_file = max(output_files, key=os.path.getctime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_completed_annotations.append(data)
        
        if all_completed_annotations:
            # 다운로드할 파일 데이터 준비
            final_json_string = json.dumps(all_completed_annotations, ensure_ascii=False, indent=4)
            
            # st.download_button 사용
            st.download_button(
                label="✅ 다운로드 준비 완료! 클릭하세요.",
                data=final_json_string,
                file_name=f"completed_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.error("완료된 파일에 해당하는 결과 파일(`output` 폴더)을 찾을 수 없습니다.")