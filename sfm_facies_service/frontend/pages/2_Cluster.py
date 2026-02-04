import os
import time
import streamlit as st
import requests
import numpy as np
import plotly.express as px

try:
    API = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
except Exception:
    API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Cluster")
st.title("Cluster embeddings")
st.caption(f"API: {API}")

resp = requests.get(f"{API}/embeddings").json()
embeddings = resp.get("embeddings", [])

if not embeddings:
    st.info("No embeddings available. Create one in the Embed page.")
    st.stop()

names = [e.get("embed_id") for e in embeddings]
embed_id = st.selectbox("Embedding", names)
meta = next((e for e in embeddings if e.get("embed_id") == embed_id), {})

st.subheader("Embedding meta")
st.json(meta)

n_clusters = st.slider("n_clusters", 2, 20, 8)
max_samples_per_slice = st.number_input("max_samples_per_slice", min_value=500, value=5000, step=500)

run_clicked = st.button("Run clustering")
if run_clicked:
    params = dict(n_clusters=int(n_clusters), max_samples_per_slice=int(max_samples_per_slice))
    r = requests.post(f"{API}/cluster_async/{embed_id}", params=params, timeout=10)
    if r.status_code != 200:
        st.error(f"Cluster failed: HTTP {r.status_code}")
        st.code(r.text)
    else:
        st.success("Clustering queued.")
        progress = st.progress(0, text="Queued")
        for _ in range(600):
            status = requests.get(f"{API}/embed_status/{embed_id}").json()
            pct = int(status.get("progress", 0))
            detail = status.get("detail", status.get("status", ""))
            progress.progress(min(max(pct, 0), 100), text=f"{pct}% - {detail}")
            if status.get("detail") == "cluster_done":
                break
            if status.get("status") == "failed":
                st.error(f"Cluster failed: {status}")
                break
            time.sleep(0.5)

cluster_meta = requests.get(f"{API}/cluster_meta/{embed_id}")
if cluster_meta.status_code == 200:
    st.subheader("Cluster meta")
    st.json(cluster_meta.json())

    pts = requests.get(f"{API}/cluster_points/{embed_id}", params={"max_points": 50000})
    if pts.status_code == 200:
        d = pts.json()
        if len(d.get("twt", [])) > 0:
            fig3d = px.scatter_3d(
                x=d["iline"],
                y=d["xline"],
                z=d["twt"],
                color=[str(v) for v in d["label"]],
                opacity=0.6,
            )
            fig3d.update_layout(scene=dict(zaxis=dict(autorange="reversed")))
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.warning("No points to display in 3D.")

    dims = meta.get("dims", {})
    axis = meta.get("slice_mode", "time")
    idx_resp = requests.get(f"{API}/cluster_indices/{embed_id}")
    idxs = []
    if idx_resp.status_code == 200:
        idxs = idx_resp.json().get("indices", [])
    if not idxs:
        st.warning("No cluster labels available. Run clustering first.")
    else:
        if len(idxs) == 1:
            idx = idxs[0]
            st.info(f"Only one slice available for {axis}: index {idx}")
        else:
            idx = st.select_slider(f"{axis} index", options=idxs, value=idxs[0])
        sl = requests.get(f"{API}/cluster_slice/{embed_id}", params={"axis": axis, "index": idx})
        if sl.status_code == 200:
            payload = sl.json()
            raw2d = np.array(payload["raw"], dtype=np.float32)
            lab2d = np.array(payload["labels"], dtype=np.int32)
            palette = [
                (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
                (188, 189, 34), (23, 190, 207),
            ]
            lab_rgb = np.zeros((*lab2d.shape, 3), dtype=np.uint8)
            valid = lab2d >= 0
            if valid.any():
                flat = lab2d[valid]
                colors = np.array([palette[v % len(palette)] for v in flat], dtype=np.uint8)
                lab_rgb[valid] = colors
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Raw slice")
                fig_raw = px.imshow(raw2d, origin="upper", aspect="equal", color_continuous_scale="Viridis")
                fig_raw.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_raw, use_container_width=True)
            with c2:
                st.markdown("### Cluster slice")
                fig_lab = px.imshow(lab_rgb, origin="upper", aspect="equal")
                st.plotly_chart(fig_lab, use_container_width=True)
        else:
            st.warning("Slice not available.")
