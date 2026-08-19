from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from machine_bias_reproduction.plates import bar_layout, shaded
from machine_bias_reproduction.questions import Question

from .matching import countries_for, survey_shares
from .palette import arm_order, arm_tone
from .plate import (
    MIXTRAL_INK,
    RULE_DASH,
    ModeFigure,
    answer_rule,
    empty_panel,
    per_mode,
)


def response_shift(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> list[Path]:
    columns = list(question.answer_columns)
    labels = [f"{code}. {name}" for code, name in zip(columns, question.wvs_labels, strict=True)]
    cultures = arm_order(
        culture for culture, tables in frames.items() if tables["responses"] is not None
    )
    positions = np.arange(len(columns))
    width, offsets = bar_layout(len(cultures))

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        wvs: np.ndarray | None = None
        drawn: list[str] = []
        for index, culture in enumerate(cultures):
            responses = frames[culture]["responses"]
            selected = responses[responses["method"] == mode]
            if selected.empty:
                continue
            drawn.append(culture)
            axis.bar(
                positions + offsets[index],
                selected.iloc[0][columns].to_numpy(dtype=np.float64),
                width=width,
                label=culture,
                color=arm_tone(culture).fill,
                edgecolor=arm_tone(culture).ink,
                linewidth=0.7,
            )
            if wvs is None:
                wvs_row = responses[responses["method"] == "WVS"]
                if not wvs_row.empty:
                    wvs = wvs_row.iloc[0][columns].to_numpy(dtype=np.float64)
        if not drawn:
            empty_panel(axis, f"no {mode} answers")
            return figure, [axis]
        if wvs is not None:
            answer_rule(axis, positions, wvs, color=MIXTRAL_INK, label="WVS")
        for culture in drawn:
            countries = countries_for(culture)
            home = survey_shares(question, countries) if countries else None
            if home is None:
                continue
            answer_rule(
                axis,
                positions,
                home,
                color=shaded(arm_tone(culture).ink),
                label=f"WVS-{culture}",
                linestyle=RULE_DASH,
            )
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Share of answers")
        axis.legend(ncol=2)
        return figure, [axis]

    return per_mode(build, "fig_culture_response_shift", destination, align_axis="y")
