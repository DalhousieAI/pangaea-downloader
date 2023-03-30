"""Functions for plotting and Exploratory Data Analysis."""

import cv2
import matplotlib.cm
import matplotlib.colors
import numpy as np
from matplotlib.pyplot import get_cmap
from sklearn.neighbors import KernelDensity

from . import requesting


def url_from_doi(doi: str) -> str:
    """
    Convert Pangaea dataset URL from src DOI format to target format.

    Source DOI format: https://doi.org/10.1594/PANGAEA.{dataset_id},
    Target URL format: https://doi.pangaea.de/10.1594/PANGAEA.{dataset_id}.
    """
    # Already in desired format
    if ".pangaea.de" in doi:
        return doi
    # Convert to desired format
    start, end = doi.split(".org")
    full = start + ".pangaea.de" + end
    return full


def img_from_url(url: str, verbose=False) -> np.array:
    """Take an image url and return retrieved image array."""
    success = False
    while not success:
        resp = requesting.get_request_with_backoff(url, stream=True)
        print(f"status code: {resp.status_code}") if verbose else 0
        success = True if (resp.status_code == 200) else False
        if success:
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img


def make_transparent_cmap() -> matplotlib.colors.ListedColormap:
    """Return a colormap with a transparent channel based on the matplotlib.cm.Reds colormap."""
    # src: https://stackoverflow.com/a/37334212/1595060
    cmap = get_cmap("Reds")
    my_cm = cmap(np.arange(cmap.N))
    my_cm[:, -1] = np.linspace(0, 1, cmap.N)
    my_cm = matplotlib.colors.ListedColormap(my_cm)
    return my_cm


def rgb_white2alpha(
    rgb, ensure_increasing=False, ensure_linear=False, lsq_linear=False
):
    """
    Convert a set of RGB colors to RGBA with maximum transparency.

    The transparency is maximised for each color individually, assuming
    that the background is white.

    Parameters
    ----------
    rgb : array_like shaped (N, 3)
        Original colors.
    ensure_increasing : bool, default=False
        Ensure that alpha values increase monotonically.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.

    Returns
    -------
    rgba : numpy.ndarray shaped (N, 4)
        Colors with maximum possible transparency, assuming a white
        background.
    """
    # The most transparent alpha we can use is given by the min of RGB
    # Convert it from saturation to opacity
    alpha = 1.0 - np.min(rgb, axis=1)
    if lsq_linear:
        # Make a least squares fit for alpha
        indices = np.arange(len(alpha))
        A = np.stack([indices, np.ones_like(indices)], axis=-1)
        m, c = np.linalg.lstsq(A, alpha, rcond=None)[0]
        # Use our least squares fit to generate a linear alpha
        alpha = c + m * indices
        alpha = np.clip(alpha, 0, 1)
    elif ensure_linear:
        # Use a linearly increasing/decreasing alpha from start to finish
        alpha = np.linspace(alpha[0], alpha[-1], rgb.shape[0])
    elif ensure_increasing:
        # Let's also ensure the alpha value is monotonically increasing
        a_max = alpha[0]
        for i, a in enumerate(alpha):
            alpha[i] = a_max = np.maximum(a, a_max)
    alpha = np.expand_dims(alpha, -1)
    # Rescale colors to discount the white that will show through from transparency
    rgb = rgb + alpha - 1
    rgb = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=(alpha > 0))
    rgb = np.clip(rgb, 0, 1)
    # Concatenate our alpha channel
    rgba = np.concatenate((rgb, alpha), axis=1)
    return rgba


def cmap_white2alpha(
    name, ensure_increasing=False, ensure_linear=False, lsq_linear=False, register=True
):
    """
    Add as much transparency as possible to a colormap, assuming white background.

    See https://stackoverflow.com/a/68809469/1960959

    Parameters
    ----------
    name : str
        Name of builtin (or registered) colormap.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.
    register : bool, default=True
        Whether to register the new colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with alpha set as low as possible.
    """
    # Fetch the cmap callable
    cmap = matplotlib.cm.get_cmap(name)
    # Get the colors out from the colormap LUT
    rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
    # Convert white to alpha
    rgba = rgb_white2alpha(
        rgb,
        ensure_increasing=ensure_increasing,
        ensure_linear=ensure_linear,
        lsq_linear=lsq_linear,
    )
    # Create a new Colormap object
    new_name = name + "_white2alpha"
    cmap_alpha = matplotlib.colors.ListedColormap(rgba, name=new_name)
    if register:
        matplotlib.cm.register_cmap(name=new_name, cmap=cmap_alpha)
    return cmap_alpha


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
    """Perform 2d kernel density estimate on longitude, latitude metadata."""
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
