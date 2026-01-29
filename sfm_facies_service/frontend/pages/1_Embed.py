import os
import time
import streamlit as st
import requests

try:
    API = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
except Exception:
    API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Embed")
st.title("Embed xarray cubes")
st.caption(f"API: {API}")

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("NetCDF (.nc)", type=["nc"])

job_id = st.session_state.get("job_id", None)
info = None

def _do_upload():
    if uploaded is None:
        st.error("No file selected.")
        return None
    try:
        r = requests.post(f"{API}/upload", files={"file": ("cube.nc", uploaded.getvalue())})
        r.raise_for_status()
        new_job_id = r.json()["job_id"]
        st.session_state["job_id"] = new_job_id
        st.success(f"Uploaded. job_id = {new_job_id}")
        return new_job_id
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

if uploaded is not None and st.sidebar.button("Upload"):
    job_id = _do_upload()

if job_id:
    info = requests.get(f"{API}/info/{job_id}").json()
    st.sidebar.markdown("### Cube info")
    st.sidebar.json(info)

st.sidebar.header("Embedding params")
model_size = st.sidebar.selectbox("Model size", ["base", "large"])
tile_size = st.sidebar.selectbox("Tile size", [224, 512])
device = st.sidebar.selectbox("Device", ["cuda", "cpu"])

twt_step = st.sidebar.number_input("twt_step", min_value=1, value=1, step=1)
iline_step = st.sidebar.number_input("iline_step", min_value=1, value=1, step=1)
xline_step = st.sidebar.number_input("xline_step", min_value=1, value=1, step=1)
slice_mode = st.sidebar.selectbox("Slice mode", ["time", "inline", "xline"])
batch_size = st.sidebar.number_input("batch_size", min_value=1, value=1, step=1)

run_clicked = st.sidebar.button("Run embedding")

if run_clicked and not job_id:
    if uploaded is not None:
        st.warning("No job_id yet. Uploading the selected file now...")
        job_id = _do_upload()
    if not job_id:
        st.error("No job_id. Upload the .nc and click Upload first.")

if run_clicked and job_id:
    params = dict(
        model_size=model_size,
        tile_size=tile_size,
        device=device,
        slice_mode=slice_mode,
        twt_step=int(twt_step),
        iline_step=int(iline_step),
        xline_step=int(xline_step),
        batch_size=int(batch_size),
    )
    try:
        r = requests.post(f"{API}/embed_async/{job_id}", params=params, timeout=10)
        if r.status_code != 200:
            st.error(f"Embed failed: HTTP {r.status_code}")
            st.code(r.text)
        else:
            embed_id = r.json().get("embed_id")
            st.success(f"Embedding queued: {embed_id}")
            st.json({"embed_id": embed_id, "params": params})
            progress = st.progress(0, text="Queued")
            for _ in range(600):
                status = requests.get(f"{API}/embed_status/{embed_id}").json()
                pct = int(status.get("progress", 0))
                detail = status.get("detail", status.get("status", ""))
                progress.progress(min(max(pct, 0), 100), text=f"{pct}% - {detail}")
                if status.get("status") == "done":
                    break
                if status.get("status") == "failed":
                    st.error(f"Embed failed: {status}")
                    break
                time.sleep(0.5)
    except Exception as e:
        st.error(f"Embed failed: {e}")

st.markdown("---")
st.subheader("Embedding history")
try:
    items = requests.get(f"{API}/embeddings").json().get("embeddings", [])
    if items:
        st.json(items)
    else:
        st.info("No embeddings yet.")
except Exception as e:
    st.error(f"Failed to load embeddings: {e}")
