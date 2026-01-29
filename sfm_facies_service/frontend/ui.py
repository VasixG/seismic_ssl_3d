import os
import time
import streamlit as st
import requests
import numpy as np
import plotly.express as px

# Streamlit raises if secrets.toml is missing; fall back to env/default.
try:
    API = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
except Exception:
    API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="SFM Facies Clustering")

st.title("Unsupervised Facies / Geometry Clustering (SFM Encoder)")
st.caption(f"API: {API}")

# -------- Upload --------
st.sidebar.header("1) Upload xarray (.nc)")
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

# -------- Params --------
st.sidebar.header("2) Model + Slice + Clustering")
model_size = st.sidebar.selectbox("Model size", ["base", "large"])
tile_size = st.sidebar.selectbox("Tile size", [224, 512])
device = st.sidebar.selectbox("Device", ["cuda", "cpu"])

try:
    models_resp = requests.get(f"{API}/models")
    models_resp.raise_for_status()
    available_models = models_resp.json().get("available", {})
except Exception:
    available_models = {}

if available_models:
    tiles_avail = available_models.get(model_size, {})
    if str(tile_size) not in tiles_avail:
        st.sidebar.warning("No weights for selected model size / tile size")

slice_mode = st.sidebar.selectbox("Slice mode", ["time", "window_agg", "inline", "xline"])

twt = None
twt_lo = None
twt_hi = None
agg = "rms"
iline = None
xline = None

if slice_mode == "time":
    twt = st.sidebar.number_input("twt (ms)", value=float(info["twt_min"]) if info else 0.0)
elif slice_mode == "window_agg":
    twt_lo = st.sidebar.number_input("twt_lo", value=float(info["twt_min"]) if info else 0.0)
    twt_hi = st.sidebar.number_input("twt_hi", value=float(info["twt_max"]) if info else 0.0)
    agg = st.sidebar.selectbox("Aggregation", ["rms", "mean_abs", "mean", "std"])
elif slice_mode == "inline":
    iline = st.sidebar.number_input("iline", value=int(info["iline_min"]) if info else 0)
elif slice_mode == "xline":
    xline = st.sidebar.number_input("xline", value=int(info["xline_min"]) if info else 0)

n_clusters = st.sidebar.slider("n_clusters", 2, 20, 8)
alpha = st.sidebar.slider("overlay alpha", 0.0, 1.0, 0.45)

# -------- Volume Params --------
st.sidebar.header("3) Volume (3D)")
include_time = st.sidebar.checkbox("Include time slices", value=True)
include_inline = st.sidebar.checkbox("Include inline slices", value=True)
include_xline = st.sidebar.checkbox("Include xline slices", value=True)
twt_step = st.sidebar.number_input("twt_step", min_value=1, value=1, step=1)
iline_step = st.sidebar.number_input("iline_step", min_value=1, value=1, step=1)
xline_step = st.sidebar.number_input("xline_step", min_value=1, value=1, step=1)
max_samples_per_slice = st.sidebar.number_input("max_samples_per_slice", min_value=500, value=5000, step=500)

# -------- Run --------
run_clicked = st.sidebar.button("Run features + clustering")

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
        n_clusters=n_clusters,
    )
    if twt is not None:
        params["twt"] = twt
    if twt_lo is not None:
        params["twt_lo"] = twt_lo
    if twt_hi is not None:
        params["twt_hi"] = twt_hi
    if agg is not None:
        params["agg"] = agg
    if iline is not None:
        params["iline"] = iline
    if xline is not None:
        params["xline"] = xline

    try:
        r = requests.post(f"{API}/run_async/{job_id}", params=params, timeout=10)
        if r.status_code != 200:
            st.error(f"Run failed: HTTP {r.status_code}")
            st.code(r.text)
        else:
            st.success("Job queued.")
            st.json({"job_id": job_id, "params": params})
            progress = st.progress(0, text="Queued")
            for _ in range(300):
                status = requests.get(f"{API}/status/{job_id}").json()
                pct = int(status.get("progress", 0))
                detail = status.get("detail", status.get("status", ""))
                progress.progress(min(max(pct, 0), 100), text=f"{pct}% - {detail}")
                if status.get("status") == "done":
                    break
                if status.get("status") == "failed":
                    st.error(f"Run failed: {status}")
                    break
                time.sleep(0.5)
            res = requests.get(f"{API}/result/{job_id}")
            if res.status_code == 200:
                st.session_state["run_meta"] = res.json()
                st.success("Done!")
            else:
                st.warning("Run finished, but result not found yet.")
    except Exception as e:
        st.error(f"Run failed: {e}")

# -------- Run 3D Volume --------
run_volume_clicked = st.sidebar.button("Run 3D volume clustering")

if run_volume_clicked and not job_id:
    if uploaded is not None:
        st.warning("No job_id yet. Uploading the selected file now...")
        job_id = _do_upload()
    if not job_id:
        st.error("No job_id. Upload the .nc and click Upload first.")

if run_volume_clicked and job_id:
    params = dict(
        model_size=model_size,
        tile_size=tile_size,
        device=device,
        n_clusters=n_clusters,
        twt_step=int(twt_step),
        iline_step=int(iline_step),
        xline_step=int(xline_step),
        include_time=include_time,
        include_inline=include_inline,
        include_xline=include_xline,
        max_samples_per_slice=int(max_samples_per_slice),
    )
    try:
        r = requests.post(f"{API}/run_volume_async/{job_id}", params=params, timeout=10)
        if r.status_code != 200:
            st.error(f"Run failed: HTTP {r.status_code}")
            st.code(r.text)
        else:
            st.success("3D job queued.")
            st.json({"job_id": job_id, "params": params})
            progress = st.progress(0, text="Queued")
            for _ in range(600):
                status = requests.get(f"{API}/status/{job_id}").json()
                pct = int(status.get("progress", 0))
                detail = status.get("detail", status.get("status", ""))
                progress.progress(min(max(pct, 0), 100), text=f"{pct}% - {detail}")
                if status.get("status") == "done":
                    break
                if status.get("status") == "failed":
                    st.error(f"Run failed: {status}")
                    break
                time.sleep(0.5)
            meta = requests.get(f"{API}/volume_meta/{job_id}")
            if meta.status_code == 200:
                st.session_state["run_meta_volume"] = meta.json()
                st.success("3D volume ready!")
            else:
                st.warning("3D run finished, but volume meta not found yet.")
    except Exception as e:
        st.error(f"Run failed: {e}")

# -------- Display --------
run_meta = st.session_state.get("run_meta", None)
if job_id and run_meta:
    st.subheader("Run meta")
    st.json(run_meta)

    img = requests.get(f"{API}/array/{job_id}/img2d.npy").json()
    labels = requests.get(f"{API}/array/{job_id}/labels.npy").json()
    emb = requests.get(f"{API}/array/{job_id}/emb.npy").json()
    emb_labels = requests.get(f"{API}/array/{job_id}/emb_labels.npy").json()

    img2d = np.array(img["data"], dtype=np.float32)
    lab = np.array(labels["data"], dtype=np.int32)
    emb2 = np.array(emb["data"], dtype=np.float32)
    y = np.array(emb_labels["data"], dtype=np.int32)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Seismic slice")
        st.image(img2d, use_container_width=True, clamp=True)

    with col2:
        st.markdown("### Clusters (patch grid)")
        st.image(lab.astype(np.float32), use_container_width=True, clamp=True)

    st.markdown("### Embedding (PCA сейчас; легко заменить на UMAP в backend)")
    fig = px.scatter(x=emb2[:, 0], y=emb2[:, 1], color=y.astype(str), opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Overlay (upsampled to approx image scale)")
    try:
        import cv2
        h_pat, w_pat = lab.shape
        h = h_pat * 16
        w = w_pat * 16
        lab_up = cv2.resize(lab.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        lab_up = lab_up[:img2d.shape[0], :img2d.shape[1]]

        overlay = img2d.copy()
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)

        fig2 = px.imshow(overlay, origin="upper", aspect="auto")
        fig2.add_trace(px.imshow(lab_up, origin="upper").data[0])
        fig2.data[1].opacity = alpha
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Overlay requires opencv-python. Error: {e}")
else:
    st.info("Upload cube, set params, click Run.")

# -------- 3D Volume Display --------
vol_meta = st.session_state.get("run_meta_volume", None)
if job_id and vol_meta:
    st.subheader("3D Volume")
    st.json(vol_meta)

    pts = requests.get(f"{API}/volume_points/{job_id}", params={"max_points": 50000})
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

    axis = st.selectbox("Slice axis", ["twt", "iline", "xline"], index=0)
    dims = vol_meta.get("dims", {})
    max_idx = int(dims.get(axis, 1)) - 1
    idx = st.slider("Slice index", 0, max(0, max_idx), 0)
    sl = requests.get(f"{API}/volume_slice/{job_id}", params={"axis": axis, "index": idx})
    if sl.status_code == 200:
        payload = sl.json()
        raw2d = np.array(payload["raw"], dtype=np.float32)
        lab2d = np.array(payload["labels"], dtype=np.int32)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Raw slice")
            st.image(raw2d, use_container_width=True, clamp=True)
        with c2:
            st.markdown("### Cluster slice")
            st.image(lab2d.astype(np.float32), use_container_width=True, clamp=True)
    else:
        st.warning("Slice not available.")
