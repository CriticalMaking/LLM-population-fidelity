"""Distance and quality metrics: nEMD, EMD, KL, JS, MMD."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.distance import pdist, squareform

FloatArray = npt.NDArray[np.float64]
DistanceFn = Callable[[npt.ArrayLike, npt.ArrayLike], FloatArray]

QUALITY_BINS = (-np.inf, 0.05, 0.10, 0.15, 0.30, np.inf)
QUALITY_LABELS = ("Very good", "Good", "Mediocre", "Bad", "Very bad")

KL_EPSILON = 1e-9
"""Additive smoothing for KL. WVS subpopulation proportions contain exact zeros."""

MMD_BANDWIDTH = 1.0
"""RBF bandwidth in category positions, so adjacent answers are one unit apart."""


def _pair(left: npt.ArrayLike, right: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return both distributions as 2-D float arrays of the same shape."""
    left_array = np.atleast_2d(np.asarray(left, dtype=np.float64))
    right_array = np.atleast_2d(np.asarray(right, dtype=np.float64))
    if left_array.shape != right_array.shape:
        right_array = np.broadcast_to(right_array, left_array.shape)
    if left_array.shape[-1] < 2:
        raise ValueError("a distance needs at least two ordered response categories")
    return left_array, right_array


def _smoothed(array: FloatArray, epsilon: float) -> FloatArray:
    shifted = array + epsilon
    return np.asarray(shifted / shifted.sum(axis=-1, keepdims=True), dtype=np.float64)


def nemd(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    """Compute normalized ordered Earth-Mover's Distance row-wise."""
    left_array, right_array = _pair(left, right)
    cumulative = np.cumsum(left_array - right_array, axis=-1)
    values = np.abs(cumulative).sum(axis=-1) / (left_array.shape[-1] - 1)
    return np.asarray(np.squeeze(values), dtype=np.float64)


def emd(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    """Compute ordered Earth-Mover's Distance on unit-spaced categories.

    The Wasserstein-1 distance between the two distributions when category
    ``i`` sits at position ``i``. Unnormalized, so it grows with the number of
    answer options and is comparable only within one question.
    """
    left_array, right_array = _pair(left, right)
    cumulative = np.cumsum(left_array - right_array, axis=-1)
    return np.asarray(np.squeeze(np.abs(cumulative).sum(axis=-1)), dtype=np.float64)


def kl_divergence(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    *,
    epsilon: float = KL_EPSILON,
) -> FloatArray:
    """Compute KL(left || right) in bits, with additive smoothing.

    Reported as KL(WVS || model): the direction that penalizes a model for
    putting no probability where respondents actually answered.
    """
    left_array, right_array = _pair(left, right)
    p = _smoothed(left_array, epsilon)
    q = _smoothed(right_array, epsilon)
    values = (p * np.log2(p / q)).sum(axis=-1)
    return np.asarray(np.squeeze(values), dtype=np.float64)


def js_divergence(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    """Compute Jensen-Shannon divergence in bits, bounded in [0, 1].

    Symmetric and finite even where one distribution is zero, so it needs no
    smoothing.
    """
    left_array, right_array = _pair(left, right)
    mixture = 0.5 * (left_array + right_array)
    with np.errstate(divide="ignore", invalid="ignore"):
        left_term = np.where(left_array > 0, left_array * np.log2(left_array / mixture), 0.0)
        right_term = np.where(right_array > 0, right_array * np.log2(right_array / mixture), 0.0)
    values = 0.5 * left_term.sum(axis=-1) + 0.5 * right_term.sum(axis=-1)
    return np.asarray(np.squeeze(np.clip(values, 0.0, 1.0)), dtype=np.float64)


def _rbf_kernel(size: int, bandwidth: float) -> FloatArray:
    positions = np.arange(size, dtype=np.float64)
    squared = (positions[:, None] - positions[None, :]) ** 2
    return np.asarray(np.exp(-squared / (2.0 * bandwidth**2)), dtype=np.float64)


def mmd(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    *,
    bandwidth: float = MMD_BANDWIDTH,
) -> FloatArray:
    """Compute Maximum Mean Discrepancy between two answer distributions.

    Kernel mean embedding with an RBF kernel over the ordered category
    positions, so ``MMD^2 = (p - q)' K (p - q)``. Unlike the divergences this
    charges a distance for confusing *adjacent* answers less than distant ones,
    without the hard cumulative structure of nEMD.
    """
    left_array, right_array = _pair(left, right)
    kernel = _rbf_kernel(left_array.shape[-1], bandwidth)
    difference = left_array - right_array
    squared = np.einsum("ij,jk,ik->i", difference, kernel, difference)
    return np.asarray(np.squeeze(np.sqrt(np.clip(squared, 0.0, None))), dtype=np.float64)


DISTANCES: dict[str, DistanceFn] = {
    "nEMD": nemd,
    "EMD": emd,
    "KL": kl_divergence,
    "JS": js_divergence,
    "MMD": mmd,
}
"""Every reported distance. ``nEMD`` stays first: it is the paper's metric."""

PRIMARY_DISTANCE = "nEMD"


def distance_frame(left: npt.ArrayLike, right: npt.ArrayLike) -> dict[str, FloatArray]:
    """Compute every registered distance between two aligned distribution sets."""
    return {name: np.atleast_1d(function(left, right)) for name, function in DISTANCES.items()}


def pairwise_nemd(distributions: npt.ArrayLike) -> FloatArray:
    """Return condensed pairwise nEMD values for ordered distributions."""
    array = np.asarray(distributions, dtype=np.float64)
    cumulative = np.cumsum(array, axis=1)
    return np.asarray(pdist(cumulative, metric="cityblock") / (array.shape[1] - 1))


def pairwise_nemd_matrix(distributions: npt.ArrayLike) -> FloatArray:
    """Return the square pairwise nEMD distance matrix."""
    return np.asarray(squareform(pairwise_nemd(distributions)), dtype=np.float64)


def quality_classes(values: npt.ArrayLike) -> pd.Categorical:
    """Classify nEMD values with the paper's quality thresholds."""
    array = np.asarray(values, dtype=np.float64)
    return pd.cut(
        array,
        bins=QUALITY_BINS,
        labels=QUALITY_LABELS,
        include_lowest=True,
        right=True,
    )
