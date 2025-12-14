#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd


def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in df.columns:
        if df[column].dtype in ["float64", "int64"]:
            x_min = df[column].min()
            x_max = df[column].max()
            if x_max != x_min:
                normalized_df[column] = (df[column] - x_min) / (x_max - x_min)
            else:
                normalized_df[column] = 0
    return normalized_df


def min_max_normalize_skip_first_column(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in df.columns[1:]:
        if df[column].dtype in ["float64", "int64"]:
            x_min = df[column].min()
            x_max = df[column].max()
            if x_max != x_min:
                normalized_df[column] = (df[column] - x_min) / (x_max - x_min)
            else:
                normalized_df[column] = 0
    return normalized_df

