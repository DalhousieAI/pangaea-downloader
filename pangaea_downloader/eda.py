"""Functions for plotting and Exploratory Data Analysis."""

import cv2
import numpy as np
import requests
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import get_cmap
from sklearn.neighbors import KernelDensity


def img_from_url(url: str, verbose=False) -> np.array:
    """Take an image url and return retrieved image array."""
    success = False
    while not success:
        resp = requests.get(url, stream=True)
        print(f"status code: {resp.status_code}") if verbose else 0
        success = True if (resp.status_code == 200) else False
        if success:
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img


def make_transparent_cmap() -> ListedColormap:
    """Return a colormap with a transparent channel based on the matplotlib.cm.Reds colormap."""
    # src: https://stackoverflow.com/a/37334212/1595060
    cmap = get_cmap("Reds")
    my_cm = cmap(np.arange(cmap.N))
    my_cm[:, -1] = np.linspace(0, 1, cmap.N)
    my_cm = ListedColormap(my_cm)
    return my_cm


def kde_sklearn(
    x,
    y,
    metric="euclidean",
    bw="silverman",
    bw_factor=1.0,
    return_scores=True,
    full_grid=True,
    n_grid=100,
):
    if metric == "haversine":
        lon = x
        lat = y
        x = np.radians(lat)
        y = np.radians(lon)
    xy = np.stack([x, y], axis=-1)
    # Bandwidth calculation
    n_samp, n_feat = xy.shape
    if isinstance(bw, float):
        pass
    elif not isinstance(bw, str):
        raise ValueError(
            f"bw must be a float or a string, but {bw.__class__} instance was given"
        )
    elif bw.lower() == "silverman":
        bw = (n_samp * (n_feat + 2) / 4.0) ** (-1.0 / (n_feat + 4))  # silverman
    elif bw.lower() == "scott":
        bw = n_samp ** (-1.0 / (n_feat + 4))  # scott
    else:
        raise ValueError(f"Unsupported bandwidth estimator: {bw}")
    bw *= bw_factor
    print(f"bw: {bw}, metric: {metric}")
    # KDE
    kde = KernelDensity(
        bandwidth=bw,
        metric=metric,
        kernel="gaussian",
        algorithm="ball_tree",
    )
    kde.fit(xy)
    # Extent
    if not full_grid:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
    elif metric == "haversine":
        xmin = -np.pi
        xmax = np.pi
        ymin = -np.pi / 2
        ymax = np.pi / 2
    else:
        xmin = -180
        xmax = 180
        ymin = -90
        ymax = 90
    # Mesh grid
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, n_grid * 2), np.linspace(ymin, ymax, n_grid)
    )
    positions = np.stack([X.ravel(), Y.ravel()], axis=-1)
    if metric == "haversine":
        positions = positions[:, ::-1]
    # Z heights
    Z = np.reshape(kde.score_samples(positions), X.shape)
    if not return_scores:
        Z = np.exp(Z)
    return X, Y, Z
