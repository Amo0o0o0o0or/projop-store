"""Training and evaluation utilities for the Wisconsin diagnostic dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the packaged dataset; malignant cases are the positive class."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = (dataset.target == 0).astype(int).rename("malignant")
    return features, target


def _build_candidates() -> dict[str, Any]:
    return {
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_evaluate() -> dict[str, Any]:
    """Select by training-only cross-validation, then evaluate on held-out data."""
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard = []
    candidates = _build_candidates()
    for name, candidate in candidates.items():
        scores = cross_val_score(
            candidate, x_train, y_train, cv=folds, scoring="roc_auc", n_jobs=1
        )
        leaderboard.append(
            {"Model": name, "Mean CV ROC-AUC": scores.mean(), "CV SD": scores.std()}
        )

    leaderboard.sort(key=lambda row: row["Mean CV ROC-AUC"], reverse=True)
    selected_name = leaderboard[0]["Model"]
    selected_model = candidates[selected_name].fit(x_train, y_train)
    probabilities = selected_model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    importance = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "Feature": features.columns,
            "Importance": importance.importances_mean,
            "SD": importance.importances_std,
        }
    ).sort_values("Importance", ascending=False)

    return {
        "model": selected_model,
        "model_name": selected_name,
        "x_test": x_test,
        "y_test": y_test,
        "probabilities": probabilities,
        "predictions": predictions,
        "leaderboard": pd.DataFrame(leaderboard),
        "importance": importance_frame,
    }


def calculate_metrics(
    actual: pd.Series, probabilities: Any, threshold: float = 0.5
) -> dict[str, float | int]:
    """Calculate metrics with malignant cases treated as the positive class."""
    predicted = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()
    specificity = true_negative / (true_negative + false_positive)

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probabilities),
        "average_precision": average_precision_score(actual, probabilities),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }