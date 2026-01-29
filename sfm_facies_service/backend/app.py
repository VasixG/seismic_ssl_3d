import os
import math
from concurrent.futures import ThreadPoolExecutor
import xarray as xr
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from sklearn.cluster import MiniBatchKMeans

from store import new_job, save_json, save_npy, load_json, load_npy, DATA_DIR, update_status, load_status
from slicers import slice_time, slice_window_agg, slice_inline, slice_xline
from sfm import build_sfm_vit, load_sfm_checkpoint, features_for_2d
from clustering import cluster_feat_grid, embed_2d

app = FastAPI(title="SFM Facies Unsupervised Service")

os.makedirs(DATA_DIR, exist_ok=True)
MODELS_DIR = os.environ.get("SFM_MODELS_DIR", "/weights")
EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _patch_dim(n: int) -> int:
    return int(math.ceil(n / 16))


def _upsample_labels(labels2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # labels2d is (h_p, w_p) at patch resolution (stride 16)
    up = np.repeat(np.repeat(labels2d, 16, axis=0), 16, axis=1)
    return up[:out_h, :out_w]


def _list_model_files(model_size: str, tile_size: int):
    base_dir = os.path.join(MODELS_DIR, model_size, str(tile_size))
    if not os.path.isdir(base_dir):
        return []
    return sorted([f for f in os.listdir(base_dir) if f.endswith(".pth")])


def _resolve_model_path(model_size: str, tile_size: int):
    files = _list_model_files(model_size, tile_size)
    if not files:
        return None
    return os.path.join(MODELS_DIR, model_size, str(tile_size), files[0])


@app.post("/upload")
async def upload_nc(file: UploadFile = File(...)):
    job_id, path = new_job()
    nc_path = os.path.join(path, "cube.nc")
    with open(nc_path, "wb") as f:
        f.write(await file.read())
    save_json(path, "meta.json", {"job_id": job_id, "nc_path": nc_path})
    return {"job_id": job_id}


@app.get("/models")
def list_models():
    sizes = ["base", "large"]
    tiles = [224, 512]
    available = {}
    for s in sizes:
        for t in tiles:
            files = _list_model_files(s, t)
            if files:
                available.setdefault(s, {})[str(t)] = files
    return {"available": available, "models_dir": MODELS_DIR}


@app.get("/info/{job_id}")
def info(job_id: str):
    path = os.path.join(DATA_DIR, job_id)
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"error": "job not found"}, status_code=404)
    meta = load_json(path, "meta.json")
    ds = xr.open_dataset(meta["nc_path"])
    var = list(ds.data_vars)[0]
    da = ds[var]
    out = {
        "job_id": job_id,
        "var": var,
        "dims": {k: int(v) for k, v in da.sizes.items()},
        "twt_min": float(da.twt.values.min()),
        "twt_max": float(da.twt.values.max()),
        "iline_min": int(da.iline.values.min()),
        "iline_max": int(da.iline.values.max()),
        "xline_min": int(da.xline.values.min()),
        "xline_max": int(da.xline.values.max()),
    }
    ds.close()
    return out


@app.post("/run/{job_id}")
def run_pipeline(
    job_id: str,
    model_size: str = "base",
    tile_size: int = 224,
    device: str = "cuda",
    slice_mode: str = "time",
    twt: float = None,
    twt_lo: float = None,
    twt_hi: float = None,
    agg: str = "rms",
    iline: int = None,
    xline: int = None,
    n_clusters: int = 8,
):
    print(f"[run] job_id={job_id} model_size={model_size} tile_size={tile_size} device={device} slice_mode={slice_mode} n_clusters={n_clusters}")
    path = os.path.join(DATA_DIR, job_id)
    try:
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            return JSONResponse({"error": "job not found"}, status_code=404)
        meta = load_json(path, "meta.json")
        ds = xr.open_dataset(meta["nc_path"])
        update_status(path, "running", 5, "slice")
        if slice_mode == "time":
            img2d, slice_meta = slice_time(ds, twt_value=twt)
        elif slice_mode == "window_agg":
            img2d, slice_meta = slice_window_agg(ds, twt_lo=twt_lo, twt_hi=twt_hi, agg=agg)
        elif slice_mode == "inline":
            img2d, slice_meta = slice_inline(ds, iline_value=iline)
        elif slice_mode == "xline":
            img2d, slice_meta = slice_xline(ds, xline_value=xline)
        else:
            ds.close()
            return JSONResponse({"error": "bad slice_mode"}, status_code=400)

        ds.close()

        update_status(path, "running", 15, "load_model")
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                device = "cpu"

        ckpt_path = _resolve_model_path(model_size, tile_size)
        if not ckpt_path or not os.path.exists(ckpt_path):
            return JSONResponse(
                {"error": "model not found", "models_dir": MODELS_DIR, "model_size": model_size, "tile_size": tile_size},
                status_code=404,
            )

        vit = build_sfm_vit(model_size=model_size, img_size=tile_size, in_chans=1)
        vit = load_sfm_checkpoint(vit, ckpt_path, map_location="cpu")

        update_status(path, "running", 30, "features")
        feat_grid = features_for_2d(vit, img2d, tile_size=tile_size, device=device, batch_tiles=8)
        update_status(path, "running", 70, "clustering")
        labels = cluster_feat_grid(feat_grid, n_clusters=n_clusters)
        update_status(path, "running", 85, "embedding")
        emb, emb_labels = embed_2d(feat_grid, labels)

        update_status(path, "running", 95, "saving")
        save_npy(path, "img2d.npy", img2d.astype(np.float32))
        save_npy(path, "feat_grid.npy", feat_grid.astype(np.float32))
        save_npy(path, "labels.npy", labels.astype(np.int32))
        save_npy(path, "emb.npy", emb.astype(np.float32))
        save_npy(path, "emb_labels.npy", emb_labels.astype(np.int32))

        run_meta = {
            "model_size": model_size,
            "tile_size": tile_size,
            "ckpt_file": os.path.basename(ckpt_path) if ckpt_path else None,
            "device": device,
            "slice_mode": slice_mode,
            "slice_meta": slice_meta,
            "n_clusters": int(n_clusters),
            "feat_shape": list(feat_grid.shape),
            "labels_shape": list(labels.shape),
        }
        save_json(path, "run_meta.json", run_meta)

        update_status(path, "done", 100, "done")
        print(f"[run] done job_id={job_id} feat_shape={feat_grid.shape} labels_shape={labels.shape}")
        return {"job_id": job_id, "run_meta": run_meta}
    except Exception as e:
        update_status(path, "failed", 100, str(e))
        raise


@app.post("/run_async/{job_id}")
def run_pipeline_async(
    job_id: str,
    model_size: str = "base",
    tile_size: int = 224,
    device: str = "cuda",
    slice_mode: str = "time",
    twt: float = None,
    twt_lo: float = None,
    twt_hi: float = None,
    agg: str = "rms",
    iline: int = None,
    xline: int = None,
    n_clusters: int = 8,
):
    path = os.path.join(DATA_DIR, job_id)
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"error": "job not found"}, status_code=404)
    update_status(path, "queued", 0, "queued")
    EXECUTOR.submit(
        run_pipeline,
        job_id=job_id,
        model_size=model_size,
        tile_size=tile_size,
        device=device,
        slice_mode=slice_mode,
        twt=twt,
        twt_lo=twt_lo,
        twt_hi=twt_hi,
        agg=agg,
        iline=iline,
        xline=xline,
        n_clusters=n_clusters,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/run_volume_async/{job_id}")
def run_volume_async(
    job_id: str,
    model_size: str = "base",
    tile_size: int = 224,
    device: str = "cuda",
    n_clusters: int = 8,
    twt_step: int = 1,
    iline_step: int = 1,
    xline_step: int = 1,
    include_time: bool = True,
    include_inline: bool = True,
    include_xline: bool = True,
    max_samples_per_slice: int = 5000,
):
    path = os.path.join(DATA_DIR, job_id)
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"error": "job not found"}, status_code=404)
    update_status(path, "queued", 0, "queued")
    EXECUTOR.submit(
        run_volume_pipeline,
        job_id=job_id,
        model_size=model_size,
        tile_size=tile_size,
        device=device,
        n_clusters=n_clusters,
        twt_step=twt_step,
        iline_step=iline_step,
        xline_step=xline_step,
        include_time=include_time,
        include_inline=include_inline,
        include_xline=include_xline,
        max_samples_per_slice=max_samples_per_slice,
    )
    return {"job_id": job_id, "status": "queued"}


def run_volume_pipeline(
    job_id: str,
    model_size: str,
    tile_size: int,
    device: str,
    n_clusters: int,
    twt_step: int,
    iline_step: int,
    xline_step: int,
    include_time: bool,
    include_inline: bool,
    include_xline: bool,
    max_samples_per_slice: int,
):
    path = os.path.join(DATA_DIR, job_id)
    try:
        meta = load_json(path, "meta.json")
        ds = xr.open_dataset(meta["nc_path"])
        var = list(ds.data_vars)[0]
        da = ds[var]

        n_iline = int(da.sizes["iline"])
        n_xline = int(da.sizes["xline"])
        n_twt = int(da.sizes["twt"])
        iline_p = _patch_dim(n_iline)
        xline_p = _patch_dim(n_xline)

        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                device = "cpu"

        ckpt_path = _resolve_model_path(model_size, tile_size)
        if not ckpt_path or not os.path.exists(ckpt_path):
            update_status(path, "failed", 100, "model not found")
            return

        vit = build_sfm_vit(model_size=model_size, img_size=tile_size, in_chans=1)
        vit = load_sfm_checkpoint(vit, ckpt_path, map_location="cpu")

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=2048, n_init="auto")

        twt_idx = list(range(0, n_twt, max(1, twt_step))) if include_time else []
        iline_idx = list(range(0, n_iline, max(1, iline_step))) if include_inline else []
        xline_idx = list(range(0, n_xline, max(1, xline_step))) if include_xline else []

        total_slices = len(twt_idx) + len(iline_idx) + len(xline_idx)
        done = 0
        update_status(path, "running", 5, "fit")

        for t in twt_idx:
            img2d = da.isel(twt=t).values
            feat = features_for_2d(vit, img2d, tile_size=tile_size, device=device, batch_tiles=8)
            x = feat.reshape(-1, feat.shape[-1])
            if max_samples_per_slice and x.shape[0] > max_samples_per_slice:
                idx = np.random.RandomState(0).choice(x.shape[0], size=max_samples_per_slice, replace=False)
                x = x[idx]
            km.partial_fit(x)
            done += 1
            update_status(path, "running", 5 + int(45 * done / max(1, total_slices)), "fit")

        for i in iline_idx:
            img2d = da.isel(iline=i).values
            feat = features_for_2d(vit, img2d, tile_size=tile_size, device=device, batch_tiles=8)
            x = feat.reshape(-1, feat.shape[-1])
            if max_samples_per_slice and x.shape[0] > max_samples_per_slice:
                idx = np.random.RandomState(0).choice(x.shape[0], size=max_samples_per_slice, replace=False)
                x = x[idx]
            km.partial_fit(x)
            done += 1
            update_status(path, "running", 5 + int(45 * done / max(1, total_slices)), "fit")

        for j in xline_idx:
            img2d = da.isel(xline=j).values
            feat = features_for_2d(vit, img2d, tile_size=tile_size, device=device, batch_tiles=8)
            x = feat.reshape(-1, feat.shape[-1])
            if max_samples_per_slice and x.shape[0] > max_samples_per_slice:
                idx = np.random.RandomState(0).choice(x.shape[0], size=max_samples_per_slice, replace=False)
                x = x[idx]
            km.partial_fit(x)
            done += 1
            update_status(path, "running", 5 + int(45 * done / max(1, total_slices)), "fit")

        if not twt_idx:
            update_status(path, "failed", 100, "time slices are required for 3D cube labels")
            return

        labels3d = np.full((n_twt, iline_p, xline_p), -1, dtype=np.int32)
        update_status(path, "running", 55, "label_time")

        for k, t in enumerate(twt_idx):
            img2d = da.isel(twt=t).values
            feat = features_for_2d(vit, img2d, tile_size=tile_size, device=device, batch_tiles=8)
            x = feat.reshape(-1, feat.shape[-1])
            y = km.predict(x).reshape(feat.shape[0], feat.shape[1])
            labels3d[t, :y.shape[0], :y.shape[1]] = y
            update_status(path, "running", 55 + int(40 * (k + 1) / max(1, len(twt_idx))), "label_time")

        save_npy(path, "labels3d.npy", labels3d)
        run_meta = {
            "model_size": model_size,
            "tile_size": tile_size,
            "ckpt_file": os.path.basename(ckpt_path),
            "device": device,
            "n_clusters": int(n_clusters),
            "include_time": include_time,
            "include_inline": include_inline,
            "include_xline": include_xline,
            "twt_step": int(twt_step),
            "iline_step": int(iline_step),
            "xline_step": int(xline_step),
            "dims": {"twt": n_twt, "iline": n_iline, "xline": n_xline},
            "patch_dims": {"iline_p": iline_p, "xline_p": xline_p},
        }
        save_json(path, "run_meta_volume.json", run_meta)
        update_status(path, "done", 100, "done")
        ds.close()
    except Exception as e:
        update_status(path, "failed", 100, str(e))
        raise


@app.get("/result/{job_id}")
def get_result(job_id: str):
    path = os.path.join(DATA_DIR, job_id)
    run_meta_path = os.path.join(path, "run_meta.json")
    if not os.path.exists(run_meta_path):
        return JSONResponse({"error": "run not found"}, status_code=404)
    run_meta = load_json(path, "run_meta.json")
    return run_meta


@app.get("/volume_meta/{job_id}")
def volume_meta(job_id: str):
    path = os.path.join(DATA_DIR, job_id)
    meta_path = os.path.join(path, "run_meta_volume.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"error": "volume not found"}, status_code=404)
    return load_json(path, "run_meta_volume.json")


@app.get("/volume_slice/{job_id}")
def volume_slice(job_id: str, axis: str, index: int):
    path = os.path.join(DATA_DIR, job_id)
    labels_path = os.path.join(path, "labels3d.npy")
    if not os.path.exists(labels_path):
        return JSONResponse({"error": "labels not found"}, status_code=404)
    meta = load_json(path, "meta.json")
    ds = xr.open_dataset(meta["nc_path"])
    var = list(ds.data_vars)[0]
    da = ds[var]
    labels3d = load_npy(path, "labels3d.npy")

    axis = axis.lower()
    if axis == "twt":
        raw = da.isel(twt=index).values
        lab = labels3d[index]
        lab_up = _upsample_labels(lab, raw.shape[0], raw.shape[1])
        out = {"raw": raw.tolist(), "labels": lab_up.tolist()}
    elif axis == "iline":
        raw = da.isel(iline=index).values  # (xline, twt)
        iline_p = min(labels3d.shape[1] - 1, index // 16)
        lab = labels3d[:, iline_p, :].T  # (xline_p, twt)
        lab_up = _upsample_labels(lab, raw.shape[0], raw.shape[1])
        out = {"raw": raw.tolist(), "labels": lab_up.tolist()}
    elif axis == "xline":
        raw = da.isel(xline=index).values  # (iline, twt)
        xline_p = min(labels3d.shape[2] - 1, index // 16)
        lab = labels3d[:, :, xline_p].T  # (iline_p, twt)
        lab_up = _upsample_labels(lab, raw.shape[0], raw.shape[1])
        out = {"raw": raw.tolist(), "labels": lab_up.tolist()}
    else:
        ds.close()
        return JSONResponse({"error": "axis must be twt/iline/xline"}, status_code=400)

    ds.close()
    return out


@app.get("/volume_points/{job_id}")
def volume_points(job_id: str, max_points: int = 50000):
    path = os.path.join(DATA_DIR, job_id)
    labels_path = os.path.join(path, "labels3d.npy")
    if not os.path.exists(labels_path):
        return JSONResponse({"error": "labels not found"}, status_code=404)
    labels3d = load_npy(path, "labels3d.npy")
    pts = np.argwhere(labels3d >= 0)
    if pts.shape[0] == 0:
        return {"x": [], "y": [], "z": [], "label": []}
    if pts.shape[0] > max_points:
        idx = np.random.RandomState(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    twt = pts[:, 0]
    ilp = pts[:, 1]
    xlp = pts[:, 2]
    lbl = labels3d[pts[:, 0], pts[:, 1], pts[:, 2]]
    return {
        "twt": twt.tolist(),
        "iline": (ilp * 16).tolist(),
        "xline": (xlp * 16).tolist(),
        "label": lbl.tolist(),
    }


@app.get("/status/{job_id}")
def status(job_id: str):
    path = os.path.join(DATA_DIR, job_id)
    return load_status(path)


@app.get("/arrays/{job_id}")
def arrays(job_id: str):
    return {"ok": True}


@app.get("/array/{job_id}/{name}")
def get_array(job_id: str, name: str):
    path = os.path.join(DATA_DIR, job_id)
    allowed = {"img2d.npy", "labels.npy", "emb.npy", "emb_labels.npy"}
    if name not in allowed:
        return JSONResponse({"error": "not allowed"}, status_code=400)
    arr_path = os.path.join(path, name)
    if not os.path.exists(arr_path):
        return JSONResponse({"error": "array not found"}, status_code=404)
    arr = load_npy(path, name)
    return {"name": name, "shape": list(arr.shape), "data": arr.tolist()}
