from __future__ import annotations

import pytest

from machine_bias_reproduction.data import (
    MIN_VALID_ANSWERS_PER_SUBPOPULATION,
    archived_fa,
    archived_ntp,
    prepare_data,
)


@pytest.mark.integration
def test_a_complete_run_keeps_every_subpopulation() -> None:
    data = prepare_data(archived_ntp("d_happy"), archived_fa("d_happy"), "d_happy")
    assert data.coverage.complete
    assert data.coverage.subpopulations_retained == 687
    assert data.coverage.ntp_observed == data.coverage.ntp_expected == 13_904
    assert data.coverage.fa_observed == data.coverage.fa_expected == 26_981


@pytest.mark.integration
def test_a_partial_run_is_scored_on_what_it_answered() -> None:
    ntp = archived_ntp("d_happy")
    fa = archived_fa("d_happy")
    partial = prepare_data(ntp.head(4_000), fa.head(8_000), "d_happy")

    assert not partial.coverage.complete
    assert partial.coverage.ntp_observed < partial.coverage.ntp_expected
    assert partial.coverage.subpopulations_dropped > 0
    assert len(partial.names) == partial.coverage.subpopulations_retained
    assert partial.wvs_props.notna().all().all()
    assert partial.props("ntp").notna().all().all()


@pytest.mark.integration
def test_a_run_with_only_full_answers_is_scored_on_them() -> None:
    fa_only = prepare_data(None, archived_fa("d_happy"), "d_happy")

    assert fa_only.modes() == ("fa",)
    assert fa_only.ntp_props is None
    assert fa_only.coverage.ntp_expected == 0
    assert fa_only.coverage.ntp_observed == 0
    assert fa_only.coverage.fa_observed == fa_only.coverage.fa_expected == 26_981
    assert fa_only.coverage.subpopulations_retained == 687
    assert fa_only.props("fa").notna().all().all()
    with pytest.raises(ValueError):
        fa_only.props("ntp")


def test_a_run_with_neither_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        prepare_data(None, None, "d_happy")


@pytest.mark.integration
def test_a_question_nobody_answered_drops_only_the_undefined_subpopulations() -> None:
    data = prepare_data(archived_ntp("d_polpos"), archived_fa("d_polpos"), "d_polpos")
    assert data.coverage.ntp_observed == data.coverage.ntp_expected
    assert data.coverage.subpopulations_retained == 639
    assert data.coverage.subpopulations_dropped == 48


def test_the_valid_answer_floor_matches_the_subpopulation_floor() -> None:
    assert MIN_VALID_ANSWERS_PER_SUBPOPULATION == 20
