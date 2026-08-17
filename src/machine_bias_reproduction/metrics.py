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

KL_ZERO_PROPORTION_SMOOTHING = 1e-9

MMD_BANDWIDTH_IN_CATEGORY_STEPS = 1.0


def _pair(left: npt.ArrayLike, right: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
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
    left_array, right_array = _pair(left, right)
    cumulative = np.cumsum(left_array - right_array, axis=-1)
    values = np.abs(cumulative).sum(axis=-1) / (left_array.shape[-1] - 1)
    return np.asarray(np.squeeze(values), dtype=np.float64)


def emd(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    left_array, right_array = _pair(left, right)
    cumulative = np.cumsum(left_array - right_array, axis=-1)
    return np.asarray(np.squeeze(np.abs(cumulative).sum(axis=-1)), dtype=np.float64)


def kl_divergence(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    *,
    epsilon: float = KL_ZERO_PROPORTION_SMOOTHING,
) -> FloatArray:
    left_array, right_array = _pair(left, right)
    left_smoothed = _smoothed(left_array, epsilon)
    right_smoothed = _smoothed(right_array, epsilon)
    values = (left_smoothed * np.log2(left_smoothed / right_smoothed)).sum(axis=-1)
    return np.asarray(np.squeeze(values), dtype=np.float64)


def js_divergence(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
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
    bandwidth: float = MMD_BANDWIDTH_IN_CATEGORY_STEPS,
) -> FloatArray:
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


def pairwise_nemd(distributions: npt.ArrayLike) -> FloatArray:
    array = np.asarray(distributions, dtype=np.float64)
    cumulative = np.cumsum(array, axis=1)
    return np.asarray(pdist(cumulative, metric="cityblock") / (array.shape[1] - 1))


def pairwise_nemd_matrix(distributions: npt.ArrayLike) -> FloatArray:
    return np.asarray(squareform(pairwise_nemd(distributions)), dtype=np.float64)


def quality_classes(values: npt.ArrayLike) -> pd.Categorical:
    array = np.asarray(values, dtype=np.float64)
    return pd.cut(
        array,
        bins=QUALITY_BINS,
        labels=QUALITY_LABELS,
        include_lowest=True,
        right=True,
    )
