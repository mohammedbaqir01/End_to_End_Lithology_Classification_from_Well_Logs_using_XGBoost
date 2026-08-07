# Creates geological and statistical features from well logs, including washout indicators, log relationships, depth context, rolling statistics, and gradients.
import pandas as pd
import numpy as np


def add_washout_features(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    train_df["CALI_BS_DIFF"] = train_df["CALI"] - train_df["BS"]
    train_df["IS_WASHOUT"] = (train_df["CALI_BS_DIFF"] > 2).astype(int)
    return train_df


def add_ratio_features(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    eps = 1e-6

    if {"RHOB", "NPHI"}.issubset(train_df.columns):
        train_df["RHOB_NPHI_DIFF"] = train_df["RHOB"] - train_df["NPHI"]

    if {"RDEP", "RMED"}.issubset(train_df.columns):
        train_df["RES_RATIO"] = train_df["RDEP"] / (train_df["RMED"] + eps)
        train_df["LOG_RDEP"] = np.log1p(train_df["RDEP"].clip(lower=0))
        train_df["LOG_RMED"] = np.log1p(train_df["RMED"].clip(lower=0))

    if {"DTC", "RHOB"}.issubset(train_df.columns):
        train_df["DTC_RHOB_PROD"] = train_df["DTC"] * train_df["RHOB"]

    if "PEF" in train_df.columns and "RHOB" in train_df.columns:
        train_df["PEF_RHOB_RATIO"] = train_df["PEF"] / (train_df["RHOB"] + eps)

    return train_df


def add_rolling_features(
    train_df: pd.DataFrame,
    cols: list,
    well_col: str = "well_name",
    depth_col: str = "DEPTH_MD",
    windows: list = (5, 15, 31),
) -> pd.DataFrame:
    train_df = train_df.sort_values([well_col, depth_col]).copy()
    grouped = train_df.groupby(well_col)

    for col in cols:
        if col not in train_df.columns:
            continue
        for w in windows:
            train_df[f"{col}_ROLLMEAN_{w}"] = grouped[col].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            train_df[f"{col}_ROLLSTD_{w}"] = grouped[col].transform(
                lambda s: s.rolling(w, min_periods=1).std()
            )

    return train_df


def add_gradient_features(
    train_df: pd.DataFrame,
    cols: list,
    well_col: str = "well_name",
    depth_col: str = "DEPTH_MD",
) -> pd.DataFrame:
    train_df = train_df.sort_values([well_col, depth_col]).copy()
    grouped = train_df.groupby(well_col)

    for col in cols:
        if col not in train_df.columns:
            continue
        train_df[f"{col}_GRAD"] = grouped[col].transform(lambda s: s.diff().fillna(0))

    return train_df


def add_depth_features(
    train_df: pd.DataFrame, well_col: str = "well_name", depth_col: str = "DEPTH_MD"
) -> pd.DataFrame:
    train_df = train_df.copy()
    well_min = train_df.groupby(well_col)[depth_col].transform("min")
    well_max = train_df.groupby(well_col)[depth_col].transform("max")
    span = (well_max - well_min).replace(0, np.nan)
    train_df["DEPTH_REL"] = ((train_df[depth_col] - well_min) / span).fillna(0)
    return train_df


def add_group_formation_encoding(
    train_df: pd.DataFrame,
    train_reference_df: pd.DataFrame,
    cat_cols: list = ("GROUP", "FORMATION"),
) -> pd.DataFrame:
    train_df = train_df.copy()
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        freq_map = train_reference_df[col].value_counts(normalize=True)
        train_df[f"{col}_FREQ"] = train_df[col].map(freq_map).fillna(0)
    return train_df


def build_features(
    train_df: pd.DataFrame, train_reference_df: pd.DataFrame = None
) -> pd.DataFrame:
    out = train_df.copy()
    out = add_washout_features(out)
    out = add_ratio_features(out)
    out = add_depth_features(out)
    out = add_rolling_features(out, cols=["RHOB", "NPHI", "DTC", "CALI", "PEF"])
    out = add_gradient_features(out, cols=["RHOB", "NPHI", "DTC", "CALI"])

    ref_df = train_reference_df if train_reference_df is not None else out
    out = add_group_formation_encoding(out, ref_df)

    return out


def add_washout_features(test_df: pd.DataFrame) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df["CALI_BS_DIFF"] = test_df["CALI"] - test_df["BS"]
    test_df["IS_WASHOUT"] = (test_df["CALI_BS_DIFF"] > 2).astype(int)
    return test_df


def add_ratio_features(test_df: pd.DataFrame) -> pd.DataFrame:
    test_df = test_df.copy()
    eps = 1e-6

    if {"RHOB", "NPHI"}.issubset(test_df.columns):
        test_df["RHOB_NPHI_DIFF"] = test_df["RHOB"] - test_df["NPHI"]

    if {"RDEP", "RMED"}.issubset(test_df.columns):
        test_df["RES_RATIO"] = test_df["RDEP"] / (test_df["RMED"] + eps)
        test_df["LOG_RDEP"] = np.log1p(test_df["RDEP"].clip(lower=0))
        test_df["LOG_RMED"] = np.log1p(test_df["RMED"].clip(lower=0))

    if {"DTC", "RHOB"}.issubset(test_df.columns):
        test_df["DTC_RHOB_PROD"] = test_df["DTC"] * test_df["RHOB"]

    if "PEF" in test_df.columns and "RHOB" in test_df.columns:
        test_df["PEF_RHOB_RATIO"] = test_df["PEF"] / (test_df["RHOB"] + eps)

    return test_df


def add_rolling_features(
    test_df: pd.DataFrame,
    cols: list,
    well_col: str = "well_name",
    depth_col: str = "DEPTH_MD",
    windows: list = (5, 15, 31),
) -> pd.DataFrame:
    test_df = test_df.sort_values([well_col, depth_col]).copy()
    grouped = test_df.groupby(well_col)

    for col in cols:
        if col not in test_df.columns:
            continue
        for w in windows:
            test_df[f"{col}_ROLLMEAN_{w}"] = grouped[col].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            test_df[f"{col}_ROLLSTD_{w}"] = grouped[col].transform(
                lambda s: s.rolling(w, min_periods=1).std()
            )

    return test_df


def add_gradient_features(
    test_df: pd.DataFrame,
    cols: list,
    well_col: str = "well_name",
    depth_col: str = "DEPTH_MD",
) -> pd.DataFrame:
    test_df = test_df.sort_values([well_col, depth_col]).copy()
    grouped = test_df.groupby(well_col)

    for col in cols:
        if col not in test_df.columns:
            continue
        test_df[f"{col}_GRAD"] = grouped[col].transform(lambda s: s.diff().fillna(0))

    return test_df


def add_depth_features(
    test_df: pd.DataFrame, well_col: str = "well_name", depth_col: str = "DEPTH_MD"
) -> pd.DataFrame:
    test_df = test_df.copy()
    well_min = test_df.groupby(well_col)[depth_col].transform("min")
    well_max = test_df.groupby(well_col)[depth_col].transform("max")
    span = (well_max - well_min).replace(0, np.nan)
    test_df["DEPTH_REL"] = ((test_df[depth_col] - well_min) / span).fillna(0)
    return test_df


def add_group_formation_encoding(
    test_df: pd.DataFrame,
    train_reference_df: pd.DataFrame,
    cat_cols: list = ("GROUP", "FORMATION"),
) -> pd.DataFrame:
    test_df = test_df.copy()
    for col in cat_cols:
        if col not in test_df.columns:
            continue
        freq_map = train_reference_df[col].value_counts(normalize=True)
        test_df[f"{col}_FREQ"] = test_df[col].map(freq_map).fillna(0)
    return test_df


def build_features(
    test_df: pd.DataFrame, train_reference_df: pd.DataFrame = None
) -> pd.DataFrame:
    out = test_df.copy()
    out = add_washout_features(out)
    out = add_ratio_features(out)
    out = add_depth_features(out)
    out = add_rolling_features(out, cols=["RHOB", "NPHI", "DTC", "CALI", "PEF"])
    out = add_gradient_features(out, cols=["RHOB", "NPHI", "DTC", "CALI"])

    ref_df = train_reference_df if train_reference_df is not None else out
    out = add_group_formation_encoding(out, ref_df)

    return out
