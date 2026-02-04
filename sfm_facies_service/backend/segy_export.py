import json
import numpy as np
import xarray as xr
from segysak.segy import segy_writer


def _upsample_labels(labels2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    up = np.repeat(np.repeat(labels2d, 16, axis=0), 16, axis=1)
    return up[:out_h, :out_w]


def _coord_index(coords: np.ndarray, value):
    coords = np.asarray(coords)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("coords must be a non-empty 1D array")
    idx = int(np.argmin(np.abs(coords - value)))
    return idx


def _build_cluster_cube(ds: xr.Dataset, labels2d: np.ndarray, slice_meta: dict, fill_value: int = -1):
    var_name = list(ds.data_vars)[0]
    base = ds[var_name]
    data = np.full(base.shape, fill_value, dtype=np.int32)

    ilines = ds["iline"].values
    xlines = ds["xline"].values
    twts = ds["twt"].values
    mode = slice_meta.get("mode")

    if mode == "time":
        t_idx = _coord_index(twts, slice_meta["twt"])
        data[:, :, t_idx] = labels2d
    elif mode == "window_agg":
        twt_mid = 0.5 * (slice_meta["twt_lo"] + slice_meta["twt_hi"])
        t_idx = _coord_index(twts, twt_mid)
        data[:, :, t_idx] = labels2d
    elif mode == "inline":
        i_idx = _coord_index(ilines, slice_meta["iline"])
        data[i_idx, :, :] = labels2d.T
    elif mode == "xline":
        x_idx = _coord_index(xlines, slice_meta["xline"])
        data[:, x_idx, :] = labels2d.T
    else:
        raise ValueError(f"unknown slice mode: {mode}")

    out = xr.Dataset(
        {var_name: xr.DataArray(data, dims=base.dims, coords=base.coords, attrs=base.attrs)}
    )
    out.attrs = dict(ds.attrs)
    out.attrs["cluster_slice_mode"] = mode
    out.attrs["cluster_slice_meta"] = json.dumps(slice_meta, ensure_ascii=False)
    return out


def save_cluster_segy(ds: xr.Dataset, labels2d: np.ndarray, slice_meta: dict, out_path: str):
    cube = _build_cluster_cube(ds, labels2d, slice_meta, fill_value=-1)

    trace_header_map = {}
    if "iline" in cube.coords:
        trace_header_map["iline"] = 189
    if "xline" in cube.coords:
        trace_header_map["xline"] = 193
    if "cdp_x" in cube.coords:
        trace_header_map["cdp_x"] = 73
    if "cdp_y" in cube.coords:
        trace_header_map["cdp_y"] = 77

    segy_writer(cube, out_path, trace_header_map=trace_header_map)
    return out_path


def save_labels3d_segy(ds: xr.Dataset, labels3d: np.ndarray, out_path: str):
    var_name = list(ds.data_vars)[0]
    base = ds[var_name]

    if labels3d.ndim != 3:
        raise ValueError("labels3d must be 3D (twt, iline_p, xline_p)")

    n_iline = int(ds["iline"].shape[0])
    n_xline = int(ds["xline"].shape[0])
    n_twt = int(ds["twt"].shape[0])
    if labels3d.shape[0] != n_twt:
        raise ValueError("labels3d twt dimension does not match dataset")

    data = np.full((n_iline, n_xline, n_twt), -1, dtype=np.int32)
    for t in range(n_twt):
        lab2d = labels3d[t]
        if np.all(lab2d < 0):
            continue
        lab_up = _upsample_labels(lab2d, n_iline, n_xline)
        data[:, :, t] = lab_up

    cube = xr.Dataset(
        {var_name: xr.DataArray(data, dims=base.dims, coords=base.coords, attrs=base.attrs)}
    )
    cube.attrs = dict(ds.attrs)

    trace_header_map = {}
    if "iline" in cube.coords:
        trace_header_map["iline"] = 189
    if "xline" in cube.coords:
        trace_header_map["xline"] = 193
    if "cdp_x" in cube.coords:
        trace_header_map["cdp_x"] = 73
    if "cdp_y" in cube.coords:
        trace_header_map["cdp_y"] = 77

    segy_writer(cube, out_path, trace_header_map=trace_header_map)
    return out_path
