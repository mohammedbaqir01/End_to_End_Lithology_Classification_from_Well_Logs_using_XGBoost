# Maps raw lithology codes to target classes and separates complete wells into reproducible training and test datasets.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def split_dataset(df):

    df = df.sort_values(["well_name", "DEPT"]).reset_index(drop=True)

    mapping = {30000: "Sandstone", 65000: "Shale", 70000: "Limestone"}

    df["Lithology"] = df["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(mapping)

    df = df.dropna(subset=["Lithology"])

    le = LabelEncoder()
    df["Lithology"] = le.fit_transform(df["Lithology"])
    wells = df["well_name"].unique()
    rng = np.random.default_rng(42)
    wells_shuffled = rng.permutation(wells)
    n_test = int(len(wells_shuffled) * 0.2)
    RandomWells = wells_shuffled[:n_test]

    train_df = df[~df["well_name"].isin(RandomWells)]

    test_df = df[df["well_name"].isin(RandomWells)]

    return train_df, test_df
