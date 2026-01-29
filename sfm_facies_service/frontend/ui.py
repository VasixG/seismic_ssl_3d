import os
import streamlit as st

try:
    API = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
except Exception:
    API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="SFM Facies Clustering")

st.title("SFM Facies Service")
st.caption(f"API: {API}")

st.markdown(
    """
Это многстраничный UI:
- **Embed**: создать эмбеддинги и сохранить в историю
- **Cluster**: выбрать эмбеддинг из истории, кластеризовать и визуализировать 3D + сечения
"""
)

if "job_id" in st.session_state:
    st.info(f"Current job_id: {st.session_state['job_id']}")
else:
    st.info("No job_id yet. Use the Embed page to upload a cube.")
