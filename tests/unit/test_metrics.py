from __future__ import annotations

import numpy as np
import pytest

from machine_bias_reproduction.metrics import (
    nemd,
    pairwise_nemd,
    quality_classes,
)


def test_nemd_identical_distributions_are_zero() -> None:
    values = np.array([0.2, 0.5, 0.2, 0.1])
    assert float(nemd(values, values)) == 0.0


def test_nemd_opposite_endpoints_are_one() -> None:
    left = np.array([1.0, 0.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 0.0, 1.0])
    assert float(nemd(left, right)) == 1.0


def test_nemd_vectorizes_rows_and_broadcasts_center() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    center = np.array([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(nemd(rows, center), [2 / 3, 1 / 3])


def test_pairwise_nemd_returns_condensed_upper_triangle() -> None:
    distributions = np.eye(4)
    values = pairwise_nemd(distributions)
    assert len(values) == 6
    np.testing.assert_allclose(values, [1 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3, 1 / 3])


def test_quality_boundaries_match_the_paper() -> None:
    classified = quality_classes([0.0, 0.05, 0.051, 0.10, 0.15, 0.30, 0.31])
    assert classified.tolist() == [
        "Very good",
        "Very good",
        "Good",
        "Good",
        "Mediocre",
        "Bad",
        "Very bad",
    ]


def test_nemd_rejects_one_category() -> None:
    with pytest.raises(ValueError):
        nemd([1.0], [1.0])


def test_emd_matches_scipy_wasserstein_on_unit_spaced_categories() -> None:
    from scipy.stats import wasserstein_distance

    from machine_bias_reproduction.metrics import emd

    rng = np.random.default_rng(0)
    for levels in (2, 4, 7, 10):
        support = np.arange(levels)
        for _ in range(5):
            left = rng.dirichlet(np.ones(levels))
            right = rng.dirichlet(np.ones(levels))
            expected = wasserstein_distance(support, support, left, right)
            assert float(emd(left, right)) == pytest.approx(expected, abs=1e-12)


def test_emd_is_nemd_times_the_category_span() -> None:
    from machine_bias_reproduction.metrics import emd, nemd

    left = np.array([0.4, 0.3, 0.2, 0.1])
    right = np.array([0.1, 0.2, 0.3, 0.4])
    assert float(emd(left, right)) == pytest.approx(float(nemd(left, right)) * 3)


def test_every_distance_is_zero_for_identical_distributions() -> None:
    from machine_bias_reproduction.metrics import DISTANCES

    values = np.array([0.2, 0.5, 0.2, 0.1])
    for name, function in DISTANCES.items():
        assert float(function(values, values)) == pytest.approx(0.0, abs=1e-9), name


def test_kl_is_asymmetric_and_survives_exact_zeros() -> None:
    from machine_bias_reproduction.metrics import kl_divergence

    left = np.array([0.5, 0.5, 0.0, 0.0])
    right = np.array([0.0, 0.0, 0.5, 0.5])
    forward = float(kl_divergence(left, right))
    backward = float(kl_divergence(right, left))
    assert np.isfinite(forward) and forward > 0
    assert forward == pytest.approx(backward)
    assert float(kl_divergence([0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0])) > 0


def test_js_is_symmetric_and_bounded_by_one() -> None:
    from machine_bias_reproduction.metrics import js_divergence

    left = np.array([1.0, 0.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 0.0, 1.0])
    assert float(js_divergence(left, right)) == pytest.approx(1.0)
    assert float(js_divergence(left, right)) == pytest.approx(float(js_divergence(right, left)))


def test_mmd_is_non_negative_and_charges_less_for_adjacent_confusions() -> None:
    from machine_bias_reproduction.metrics import mmd

    base = np.array([1.0, 0.0, 0.0, 0.0])
    adjacent = np.array([0.0, 1.0, 0.0, 0.0])
    distant = np.array([0.0, 0.0, 0.0, 1.0])
    assert float(mmd(base, adjacent)) < float(mmd(base, distant))
    rng = np.random.default_rng(1)
    for _ in range(50):
        left = rng.dirichlet(np.ones(4))
        right = rng.dirichlet(np.ones(4))
        assert float(mmd(left, right)) >= 0.0


def test_distances_broadcast_a_single_reference_over_many_rows() -> None:
    from machine_bias_reproduction.metrics import DISTANCES

    rows = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    center = np.array([0.0, 0.0, 1.0, 0.0])
    for name, function in DISTANCES.items():
        assert np.asarray(function(rows, center)).shape == (2,), name
