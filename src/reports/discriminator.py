from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from machine_bias_reproduction.config import GLOBAL_SEED

from .load import QuestionData

TREES = 500


def _oob_accuracy(left: np.ndarray, right: np.ndarray) -> float:
    features = np.vstack([left, right])
    labels = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
    forest = RandomForestClassifier(
        n_estimators=TREES,
        oob_score=True,
        bootstrap=True,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    forest.fit(features, labels)
    return float(forest.oob_score_)


def table_s8(loaded: dict[str, QuestionData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, data in loaded.items():
        reference = data.wvs.to_numpy(dtype=np.float64)
        candidates = dict(data.series)
        if data.linear is not None:
            candidates["Linear"] = data.linear
        if data.random:
            candidates["Random"] = data.random[0]
        for name, props in candidates.items():
            values = props.to_numpy(dtype=np.float64)
            if values.shape != reference.shape or not np.isfinite(values).all():
                continue
            rows.append(
                {
                    "question": var,
                    "series": name,
                    "subpopulations": len(values),
                    "oob_accuracy_percent": 100.0 * _oob_accuracy(reference, values),
                }
            )
    return pd.DataFrame(rows)
