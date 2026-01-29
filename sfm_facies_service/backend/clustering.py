import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_feat_grid(feat_grid: np.ndarray, n_clusters: int):
    h, w, d = feat_grid.shape
    x = feat_grid.reshape(-1, d)
    x = StandardScaler().fit_transform(x)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    y = km.fit_predict(x)
    return y.reshape(h, w)


def embed_2d(feat_grid: np.ndarray, labels: np.ndarray, max_points=50000):
    h, w, d = feat_grid.shape
    x = feat_grid.reshape(-1, d)
    y = labels.reshape(-1)
    n = x.shape[0]
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    xs = StandardScaler().fit_transform(xs)
    emb = PCA(n_components=2, random_state=0).fit_transform(xs)
    return emb, ys
