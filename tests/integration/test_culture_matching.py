from __future__ import annotations

import numpy as np
import pytest

from culture.matching import countries_for, survey_shares
from machine_bias_reproduction.questions import resolve_question


@pytest.mark.integration
def test_the_german_survey_shares_are_not_the_pooled_ones() -> None:
    question = resolve_question("d_trust")
    german = survey_shares(question, countries_for("german"))

    assert german is not None
    assert german.sum() == pytest.approx(1.0)
    assert german == pytest.approx(np.array([0.370877, 0.629123]), abs=5e-7)


@pytest.mark.integration
def test_every_matched_culture_restricts_the_survey_to_its_own_respondents() -> None:
    question = resolve_question("d_happy")
    pooled = np.array([0.298288, 0.537610, 0.144171, 0.019931])

    for culture in ("english", "german", "spanish"):
        shares = survey_shares(question, countries_for(culture))
        assert shares is not None
        assert shares.sum() == pytest.approx(1.0)
        assert not np.allclose(shares, pooled, atol=1e-4)


@pytest.mark.integration
def test_an_arm_without_a_home_country_has_no_survey_to_restrict_to() -> None:
    assert survey_shares(resolve_question("d_trust"), countries_for("base")) is None
