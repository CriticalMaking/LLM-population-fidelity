from __future__ import annotations

import pandas as pd
import pytest

from culture import archived_reference, fidelity
from machine_bias_reproduction.questions import resolve_question

needs_archive = pytest.mark.skipif(
    not (archived_reference.ARCHIVED_CSV / "NTP-GPT-4T-d_happy.csv").is_file()
    or not (archived_reference.ARCHIVED_CSV / "FA-GPT-3.csv").is_file()
    or not fidelity.GROUPS_TABLE.is_file(),
    reason="the archived GPT series or the shipped groups table are not staged",
)


def test_the_archived_series_keep_mixtral_label_and_name_the_rest_after_the_model() -> None:
    assert archived_reference.archived_series("Mixtral-8x7B") == "Mixtral archived"
    assert archived_reference.archived_series("GPT-4T") == "GPT-4T archived"

    identity = archived_reference.archived_identity("GPT-4T", resolve_question("d_happy"), "ntp")

    assert list(identity) == list(fidelity.IDENTITY)
    assert identity["model_key"] is None
    assert identity["arm"] is None
    assert identity["model_label"] == "GPT-4T"
    assert identity["source"] == "archived/GPT-4T"
    assert identity["mode"] == "ntp"


def test_a_model_the_archive_never_published_is_skipped_not_scored() -> None:
    question = resolve_question("d_happy")

    assert archived_reference.archived_frames("Nobody-1B", question) == (None, None)
    assert archived_reference.load_archived("Nobody-1B", question) is None
    assert archived_reference.build_archived([question], models=("Nobody-1B",)).empty


@needs_archive
def test_gpt4t_is_scored_on_ntp_alone_and_the_mixtral_rows_reproduce_the_shipped_table() -> None:
    question = resolve_question("d_happy")

    table = archived_reference.build_archived([question], models=("GPT-4T", "Mixtral-8x7B"))
    pooled = table[table["group"].eq(fidelity.POPULATION)].set_index(["series", "mode"])

    assert set(pooled.index) == {
        ("GPT-4T archived", "ntp"),
        ("Mixtral archived", "ntp"),
        ("Mixtral archived", "fa"),
    }
    assert set(table["group"]) == {fidelity.POPULATION, *fidelity.FAMILIES}
    assert list(table.columns[: len(fidelity.IDENTITY) + 2]) == [
        *fidelity.IDENTITY,
        "group",
        "level",
    ]

    shipped = pd.read_csv(fidelity.GROUPS_TABLE)
    reference = shipped[
        shipped["series"].eq("Mixtral archived")
        & shipped["question"].eq("d_happy")
        & shipped["group"].eq(fidelity.POPULATION)
    ].set_index("mode")
    for mode in ("ntp", "fa"):
        for column in ("pfs", "score_center", "n_cells"):
            assert pooled.loc[("Mixtral archived", mode), column] == pytest.approx(
                reference.loc[mode, column]
            )
