from __future__ import annotations

import numpy as np
import pandas as pd

from machine_bias_reproduction.analysis import regressions
from machine_bias_reproduction.data import load_subpops, load_wvs, social_predictors
from machine_bias_reproduction.metrics import (
    QUALITY_LABELS,
    nemd,
    pairwise_nemd,
    quality_classes,
)
from machine_bias_reproduction.questions import QUESTIONS, Question

from .load import QuestionData
from .series import BASELINES

AGE_LABELS = ("<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")


def _methods(data: QuestionData) -> list[str]:
    methods = list(data.series)
    if data.linear is not None:
        methods.append("Linear")
    return methods


def table2(loaded: dict[str, QuestionData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, data in loaded.items():
        for name, marginal in data.marginals.items():
            rows.append(
                {
                    "question": var,
                    "question_label": QUESTIONS[var].label,
                    "series": name,
                    "nEMD": float(nemd(data.wvs_marginal, marginal)),
                }
            )
    return pd.DataFrame(rows)


def _distances(data: QuestionData) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    reference = data.wvs.to_numpy(dtype=np.float64)
    for name in _methods(data):
        props = data.props(name)
        if props is None:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "method": name,
                    "subpopulation": data.names,
                    "nEMD": nemd(reference, props.to_numpy(dtype=np.float64)),
                }
            )
        )
    for index, frame in enumerate(data.random, start=1):
        rows.append(
            pd.DataFrame(
                {
                    "method": f"Random_{index:02d}",
                    "subpopulation": data.names,
                    "nEMD": nemd(reference, frame.to_numpy(dtype=np.float64)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def subpopulation_distances(loaded: dict[str, QuestionData]) -> pd.DataFrame:
    frames = []
    for var, data in loaded.items():
        frame = _distances(data)
        if frame.empty:
            continue
        frame.insert(0, "question", var)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table3(distances: pd.DataFrame) -> pd.DataFrame:
    if distances.empty:
        return pd.DataFrame()
    frame = distances.copy()
    frame["quality"] = quality_classes(frame["nEMD"].to_numpy())
    frame["series"] = frame["method"].str.replace(r"^Random_\d+$", "Random", regex=True)
    counts = (
        frame.groupby(["question", "method", "series", "quality"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(["question", "method"], observed=False)["count"].transform("sum")
    counts["percent"] = 100.0 * counts["count"] / totals
    return (
        counts.groupby(["question", "series", "quality"], observed=False)["percent"]
        .mean()
        .reset_index()
        .pivot(index=["question", "series"], columns="quality", values="percent")
        .reindex(columns=list(QUALITY_LABELS))
        .reset_index()
    )


def table4(loaded: dict[str, QuestionData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, data in loaded.items():
        for name in ["WVS", *_methods(data)]:
            props = data.props(name)
            if props is None:
                continue
            values = pairwise_nemd(props.to_numpy(dtype=np.float64))
            rows.append(
                {
                    "question": var,
                    "series": name,
                    "pairs": int(values.size),
                    "median_pairwise_nEMD": float(np.median(values)),
                    "mean_pairwise_nEMD": float(np.mean(values)),
                }
            )
    return pd.DataFrame(rows)


def _fit_all(loaded: dict[str, QuestionData]) -> dict[tuple[str, str], object]:
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    predictors_all = social_predictors(wvs, subpops)
    fits: dict[tuple[str, str], object] = {}
    for var, data in loaded.items():
        predictors = predictors_all.reindex(data.names)
        for name, props in data.series.items():
            fits[(var, name)] = regressions(
                data.names, data.wvs, props, predictors, name.split("-", 1)[0].lower()
            )
    return fits


def table5_and_6(
    loaded: dict[str, QuestionData],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows: list[pd.DataFrame] = []
    f_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    for (var, name), artifacts in _fit_all(loaded).items():
        for frame, sink in (
            (artifacts.fit, fit_rows),  # type: ignore[attr-defined]
            (artifacts.f_tests, f_rows),  # type: ignore[attr-defined]
        ):
            tagged = frame.copy()
            tagged.insert(0, "series", name)
            tagged.insert(0, "question", var)
            sink.append(tagged)
        for model_name, frame in (
            ("social", artifacts.social_coefficients),  # type: ignore[attr-defined]
            ("full", artifacts.full_coefficients),  # type: ignore[attr-defined]
            ("full_standardized", artifacts.standardized_coefficients),  # type: ignore[attr-defined]
        ):
            tagged = frame.copy()
            tagged["model"] = model_name
            tagged.insert(0, "series", name)
            tagged.insert(0, "question", var)
            coefficient_rows.append(tagged)
    empty = pd.DataFrame()
    return (
        pd.concat(fit_rows, ignore_index=True) if fit_rows else empty,
        pd.concat(f_rows, ignore_index=True) if f_rows else empty,
        pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else empty,
    )


def table_s1() -> pd.DataFrame:
    wvs = load_wvs()
    return (
        wvs.groupby(["i_country", "i_surveyyear"]).size().rename("respondents").reset_index()
    ).rename(columns={"i_country": "country", "i_surveyyear": "survey_year"})


def table_s2() -> pd.DataFrame:
    wvs = load_wvs()
    age = pd.cut(wvs["i_age"], bins=(10, 25, 35, 45, 55, 65, 75, 999), labels=list(AGE_LABELS))
    frame = pd.DataFrame(
        {
            "country": wvs["i_country"],
            "Sex": wvs["i_sex"],
            "Age": age,
            "Education": wvs["i_education"],
            "Employment": wvs["i_employment"],
            "Marstat": wvs["i_marstat"],
        }
    )
    rows: list[dict[str, object]] = []
    for country, block in frame.groupby("country"):
        for variable in ("Sex", "Age", "Education", "Employment", "Marstat"):
            shares = block[variable].value_counts(normalize=True, dropna=True)
            for level, share in shares.items():
                rows.append(
                    {
                        "country": country,
                        "variable": variable,
                        "level": level,
                        "percent": 100.0 * float(share),
                        "n": int(block[variable].notna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def table_s4(loaded: dict[str, QuestionData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, data in loaded.items():
        for name, mass in data.ntp_mass.items():
            rows.append({"question": var, "series": name, "mean_valid_answer_mass": mass})
    return pd.DataFrame(rows)


def response_distributions(loaded: dict[str, QuestionData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, data in loaded.items():
        question: Question = data.question
        rows.append(
            {
                "question": var,
                "series": "WVS",
                **dict(zip(question.answer_columns, data.wvs_marginal, strict=True)),
            }
        )
        for name, marginal in data.marginals.items():
            rows.append(
                {
                    "question": var,
                    "series": name,
                    **dict(zip(question.answer_columns, marginal, strict=True)),
                }
            )
    return pd.DataFrame(rows)


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(value))


def to_latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    if frame.empty:
        return f"% {label}: no rows\n"
    columns = " & ".join(_escape(column) for column in frame.columns)
    body = " \\\\\n".join(
        " & ".join(_escape(value) for value in row) for row in frame.itertuples(index=False)
    )
    return (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{'l' * len(frame.columns)}}}\n"
        "\\toprule\n"
        f"{columns} \\\\\n"
        "\\midrule\n"
        f"{body} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


__all__ = [
    "BASELINES",
    "response_distributions",
    "subpopulation_distances",
    "table2",
    "table3",
    "table4",
    "table5_and_6",
    "table_s1",
    "table_s2",
    "table_s4",
    "to_latex",
]
