import numpy as np
import xarray as xr


def get_var(ds: xr.Dataset):
    return list(ds.data_vars)[0]


def slice_time(ds: xr.Dataset, twt_value=None):
    var = get_var(ds)
    da = ds[var]
    if twt_value is None:
        twt_value = float(da.twt.values[len(da.twt) // 2])
    sl = da.sel(twt=twt_value, method="nearest")
    return sl.values, {"mode": "time", "twt": float(sl.twt.values)}


def slice_window_agg(ds: xr.Dataset, twt_lo=None, twt_hi=None, agg="rms"):
    var = get_var(ds)
    da = ds[var]
    if twt_lo is None:
        twt_lo = float(da.twt.values[len(da.twt) // 3])
    if twt_hi is None:
        twt_hi = float(da.twt.values[2 * len(da.twt) // 3])

    win = da.sel(twt=slice(twt_lo, twt_hi)).values.astype(np.float32)

    if agg == "rms":
        img = np.sqrt(np.mean(win * win, axis=2) + 1e-12)
    elif agg == "mean_abs":
        img = np.mean(np.abs(win), axis=2)
    elif agg == "mean":
        img = np.mean(win, axis=2)
    elif agg == "std":
        img = np.std(win, axis=2)
    else:
        raise ValueError("agg must be one of: rms, mean_abs, mean, std")

    return img, {"mode": "window_agg", "twt_lo": float(twt_lo), "twt_hi": float(twt_hi), "agg": agg}


def slice_inline(ds: xr.Dataset, iline_value=None):
    var = get_var(ds)
    da = ds[var]
    if iline_value is None:
        iline_value = int(da.iline.values[len(da.iline) // 2])
    sl = da.sel(iline=iline_value, method="nearest")
    return sl.values.T, {"mode": "inline", "iline": int(sl.iline.values)}


def slice_xline(ds: xr.Dataset, xline_value=None):
    var = get_var(ds)
    da = ds[var]
    if xline_value is None:
        xline_value = int(da.xline.values[len(da.xline) // 2])
    sl = da.sel(xline=xline_value, method="nearest")
    return sl.values.T, {"mode": "xline", "xline": int(sl.xline.values)}
