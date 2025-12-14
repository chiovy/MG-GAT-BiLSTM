#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_comparison(
    normal_values: Union[pd.Series, pd.DataFrame],
    anomaly_values: Union[pd.Series, pd.DataFrame],
    output_path: Union[str, Path],
    title: Optional[str] = None,
    ylabel: str = "Value",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if isinstance(normal_values, pd.DataFrame):
        normal_series = normal_values.iloc[:, 0]
    else:
        normal_series = normal_values

    if isinstance(anomaly_values, pd.DataFrame):
        anomaly_series = anomaly_values.iloc[:, 0]
    else:
        anomaly_series = anomaly_values

    if len(normal_series) != len(anomaly_series):
        raise ValueError("normal_values and anomaly_values must have the same length")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 4))
    plt.plot(anomaly_series.values, label="Anomaly", color="red", linestyle="--")
    plt.plot(normal_series.values, label="Normal", color="blue")
    if title is not None:
        plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(False)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()

