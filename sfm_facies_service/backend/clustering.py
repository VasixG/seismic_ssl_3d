import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_feat_grid(feat_grid: np.ndarray, n_clusters: int, mask: np.ndarray | None = None):
    h, w, d = feat_grid.shape
    x = feat_grid.reshape(-1, d)
    if mask is not None:
        mask_flat = mask.reshape(-1).astype(bool)
        if mask_flat.shape[0] != x.shape[0]:
            raise ValueError("mask shape must match feat_grid spatial dims")
        x = x[mask_flat]
    if x.shape[0] == 0:
        return np.full((h, w), -1, dtype=np.int32)
    x = StandardScaler().fit_transform(x)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    y = km.fit_predict(x)
    if mask is None:
        return y.reshape(h, w)
    out = np.full((h * w,), -1, dtype=np.int32)
    out[mask_flat] = y
    return out.reshape(h, w)


def embed_2d(feat_grid: np.ndarray, labels: np.ndarray, max_points=50000):
    h, w, d = feat_grid.shape
    x = feat_grid.reshape(-1, d)
    y = labels.reshape(-1)
    valid = y >= 0
    if np.any(valid):
        x = x[valid]
        y = y[valid]
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    xs = StandardScaler().fit_transform(xs)
    emb = PCA(n_components=2, random_state=0).fit_transform(xs)
    return emb, ys
