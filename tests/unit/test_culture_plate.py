from __future__ import annotations

from pathlib import Path

import matplotlib

from culture import plate
from culture.palette import MODES

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def test_per_mode_writes_one_file_per_mode_and_nothing_else(tmp_path: Path) -> None:

    def build(mode: str) -> plate.ModeFigure:
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        return figure, [axis]

    produced = plate.per_mode(build, "fig_example", tmp_path)

    stems = sorted({path.stem for path in produced})
    assert stems == ["fig_example_fa", "fig_example_ntp"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fig_example_fa.pdf",
        "fig_example_fa.png",
        "fig_example_ntp.pdf",
        "fig_example_ntp.png",
    ]


def test_a_mode_with_no_data_still_gets_its_own_file(tmp_path: Path) -> None:

    def build(mode: str) -> plate.ModeFigure:
        figure, axis = plt.subplots()
        if mode == "FA":
            plate.empty_panel(axis, "no FA distances")
        else:
            axis.plot([0, 1], [0, 1])
        return figure, [axis]

    plate.per_mode(build, "fig_example", tmp_path)

    assert (tmp_path / "fig_example_fa.png").is_file()
    assert (tmp_path / "fig_example_ntp.png").is_file()


def test_the_two_files_share_one_range() -> None:

    built: dict[str, plate.ModeFigure] = {}
    for mode, limit in zip(MODES, (1.0, 5.0), strict=True):
        figure, axis = plt.subplots()
        axis.set_xlim(0.0, limit)
        built[mode] = (figure, [axis])

    plate.align(built, "x")

    assert [axes[0].get_xlim() for _, axes in built.values()] == [(0.0, 5.0), (0.0, 5.0)]
    for figure, _ in built.values():
        plt.close(figure)


def test_align_leaves_the_axes_alone_when_asked_for_nothing() -> None:

    built: dict[str, plate.ModeFigure] = {}
    for mode, limit in zip(MODES, (1.0, 5.0), strict=True):
        figure, axis = plt.subplots()
        axis.set_xlim(0.0, limit)
        built[mode] = (figure, [axis])

    plate.align(built, None)

    assert [axes[0].get_xlim() for _, axes in built.values()] == [(0.0, 1.0), (0.0, 5.0)]
    for figure, _ in built.values():
        plt.close(figure)
