"""Full pipeline runner for lithology classification from well logs.

This script demonstrates a compact end-to-end flow considered important:
- Load raw CSV well-log data from a training directory.
- Split into training and test sets (group-aware by well where applicable).
- Apply preprocessing (cleaning, encoding, scaling) to train and test.
- Build derived features consistently using the training reference.
- Train an XGBoost classifier and evaluate on the hold-out test set.

Why these parts are important:
- Data loading: ensures all available wells are read and concatenated.
- Split & grouping: keeps well-level grouping to avoid leakage across sets.
- Preprocessing & feature engineering: major impact on model quality.
- Training & evaluation: final model performance and reporting.

Adjust the data path below or make this script callable if you want to reuse
the pipeline programmatically or save artifacts (models, reports, plots).
"""

from src.data.load_data import load_data
from src.data.split_dataset import split_dataset
from src.data.preprocess_data import preprocess_data
from src.features.engineer_features import build_features
from src.modeling.train_model import train_xgboost_classifier
from src.modeling.cross_validation import cross_validate_by_well
from src.evaluation.evaluate_model import evaluate_model
from src.utils.artifact_io import save_artifact


DATA_PATH = "data/csv_data_training"

df = load_data(DATA_PATH)

# Split data into training and test sets. This should preserve well-level grouping
# to prevent data leakage between train and test.
train_df, test_df = split_dataset(df)

# Apply preprocessing to training and test sets (cleaning, encoding, scaling).
train_df, test_df = preprocess_data(train_df, test_df)


# Build features for train and test. Use the training reference to keep transforms
# consistent (e.g., statistics computed on training only).
train_df = build_features(train_df, train_reference_df=train_df)
test_df = build_features(test_df, train_reference_df=train_df)


# Prepare feature matrix X and label vector Y for training. Drop non-feature
# identifier columns and the target column from X.
X_train = train_df.drop(
    columns=[
        "Lithology",
        "well_name",
        "source_file",
        "FORCE_2020_LITHOFACIES_LITHOLOGY",
    ],
    errors="ignore",
)
y_train = train_df["Lithology"]

X_test = test_df.drop(
    columns=[
        "Lithology",
        "well_name",
        "source_file",
        "FORCE_2020_LITHOFACIES_LITHOLOGY",
    ],
    errors="ignore",
)
Y_test = test_df["Lithology"]


# Train the XGBoost classifier. The returned `model` should implement `predict()`.
model = train_xgboost_classifier(X_train, y_train)

groups_classification = train_df["well_name"]
cv_scores = cross_validate_by_well(
    X_train=X_train,
    y_train=y_train,
    groups=groups_classification,
    model=model,
    cv=4,
)

# save_artifact(model, "models/xgb_classifier.pkl")

report = evaluate_model(model, X_test, Y_test)
