# Evaluates the lithology classifier with well-grouped cross-validation so samples from the same well remain in one fold.
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, GroupKFold


def cross_validate_by_well(
    X_train,
    y_train,
    groups=None,
    model=None,
    cv=4,
):
    if model is None:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
        )

    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        groups=groups,
        cv=GroupKFold(n_splits=cv),
        scoring="f1_macro",
    )

    return scores
