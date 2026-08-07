# Configures and trains the XGBoost classifier used to predict lithology from engineered well-log features.
from xgboost import XGBClassifier


def train_xgboost_classifier(X_train, y_train):
    ClassificationModel = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42
    )
    ClassificationModel.fit(X_train, y_train)
    return ClassificationModel
