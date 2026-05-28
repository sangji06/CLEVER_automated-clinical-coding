"""Streamlit annotation tool for CLEVER ground-truth construction."""

import streamlit as st

st.set_page_config(
    page_title="응급실 기록지 라벨링 도구",
    page_icon="🏥",
)

st.title("🏥 응급실 기록지 KCD 코딩 도구")

st.sidebar.success("위에서 작업 페이지를 선택하세요.")

st.markdown(
    """
    **👈 사이드바에서 원하는 작업 페이지를 선택하여 시작하세요.**

    ### 이 도구는...
    - 응급실 기록지 텍스트를 기반으로 Ground Truth 데이터를 생성하기 위해 만들어졌습니다.
    - 모든 데이터는 사용자의 컴퓨터 내에서만 처리되며, 외부로 전송되지 않습니다.

    ### 각 페이지 기능:
    - **Labeling**: 기록지를 보며 진단명, 증상 등의 엔티티를 라벨링합니다.
    - **Refining**: 라벨링된 엔티티의 용어를 정제하고 중복을 제거합니다.
    - **KCD Coding**: 정제된 엔티티에 KCD 코드를 부여합니다.
    """
)
