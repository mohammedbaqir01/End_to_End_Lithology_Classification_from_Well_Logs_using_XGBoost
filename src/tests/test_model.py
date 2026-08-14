from pathlib import Path
from joblib import load
import pandas as pd

from src.data.split_dataset import split_dataset
from src.data.preprocess_data import preprocess_data
from src.features.engineer_features import build_features


MODEL_PATH = Path("models/xgb_classifier.pkl")
SAMPLE_DATA_PATH = Path("src/tests/sample_data")

model = load(MODEL_PATH)


def test_pred():

    df = pd.concat([
        pd.read_csv(file)
        for file in SAMPLE_DATA_PATH.glob("*.csv")
    ], ignore_index=True)

    df, _ = preprocess_data(df.copy(), df.copy())

    train_df, test_df = split_dataset(df)

    test_df = build_features(
        test_df,
        train_reference_df=train_df
    )

    X_test = test_df.drop(
        columns=[
            "Lithology",
            "well_name",
            "source_file",
            "FORCE_2020_LITHOFACIES_LITHOLOGY",
        ],
        errors="ignore",
    )

    results = model.predict(X_test)

    assert results is not None
    assert len(results) == len(X_test)
