#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, List

import numpy as np
import pandas as pd

try:
    from minepy import MINE

    _MINE_AVAILABLE = True
except ImportError:
    MINE = None
    _MINE_AVAILABLE = False
    from sklearn.feature_selection import mutual_info_regression


def compute_mic(
    x: Sequence[float],
    y: Sequence[float],
    alpha: float = 0.6,
    c: int = 15,
) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length")

    if _MINE_AVAILABLE:
        mine = MINE(alpha=alpha, c=c)
        mine.compute_score(x_arr, y_arr)
        return float(mine.mic())

    mi = mutual_info_regression(
        x_arr.reshape(-1, 1),
        y_arr,
        discrete_features=False,
    )
    return float(mi[0])


def select_significant_parameters(
    data: pd.DataFrame,
    target_column: str,
    threshold: float,
    alpha: float = 0.6,
    c: int = 15,
) -> List[str]:
    if target_column not in data.columns:
        raise ValueError(f"target_column {target_column} not found in data")

    y = data[target_column].values
    X = data.drop(columns=[target_column])
    X_mat = X.values
    columns = X.columns

    def _feature_mic(x: np.ndarray) -> float:
        return compute_mic(x, y, alpha=alpha, c=c)

    mic_values = np.apply_along_axis(_feature_mic, axis=0, arr=X_mat)
    mic_df = pd.DataFrame({"parameter": columns, "mic": mic_values})
    selected = mic_df[mic_df["mic"] >= threshold]["parameter"].tolist()
    return list(selected)

