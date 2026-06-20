import os
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

from app.features.loan_prediction.components.data_ingestion import (
    fetch_training_data
)

from app.features.loan_prediction.utils.preprocessing import (
    preprocess_data,
    apply_feature_engineering,
    extract_rule_baseline_features
)
from app.features.loan_prediction.components.evaluation import evaluate_model


# =========================================================
# Log Loss Helper
# =========================================================
def _log_loss_score(model, X, y):
    return log_loss(y, model.predict_proba(X))


# =========================================================
# Bias/Variance vs Regularization
# Shows WHY a particular C was chosen by GridSearchCV —
# fits a fresh model per C value on the actual train/val split.
# =========================================================
def plot_bias_variance_vs_regularization(X_train, y_train, X_val, y_val, C_values):

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_error = log_loss(y_val, baseline_model.predict_proba(X_val))

    train_errors, val_errors = [], []

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        train_errors.append(_log_loss_score(model, X_train, y_train))
        val_errors.append(_log_loss_score(model, X_val, y_val))

    lambda_values = [1 / c for c in C_values]

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, train_errors, marker="o", label="Training Error")
    plt.plot(lambda_values, val_errors, marker="o", label="Validation Error")
    plt.axhline(baseline_error, linestyle="--", linewidth=2, label=f"ZeroR Baseline = {baseline_error:.4f}")
    plt.xscale("log")
    plt.xlabel("Lambda (1/C)")
    plt.ylabel("Log Loss")
    plt.title("Bias vs Variance — Regularization Sweep\n(why this C was selected)")
    plt.legend()
    plt.grid(True)
    plt.show()

    best_idx = int(np.argmin(val_errors))
    print("\n===== REGULARIZATION SWEEP =====")
    print(f"Baseline Error      : {baseline_error:.4f}")
    print(f"Best C by val loss  : {C_values[best_idx]}  (lambda={lambda_values[best_idx]:.4g})")
    print(f"Min Validation Loss : {val_errors[best_idx]:.4f}")


# =========================================================
# Learning Curve
# Shows whether more training data would help, using the
# actual train/val split GridSearchCV operated on.
# =========================================================
def plot_learning_curve(X_train_full, y_train_full, X_val, y_val):

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train_full, y_train_full)
    baseline_error = log_loss(y_val, baseline_model.predict_proba(X_val))

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_errors, val_errors = [], []
    total_size = len(X_train_full)

    for fraction in train_sizes:
        size = int(total_size * fraction)
        X_subset = X_train_full[:size]
        y_subset = y_train_full[:size]

        if len(np.unique(y_subset)) < 2:
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(X_subset, y_subset)
        train_errors.append(_log_loss_score(model, X_subset, y_subset))
        val_errors.append(_log_loss_score(model, X_val, y_val))

    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, marker="o", label="Training Error")
    plt.plot(val_errors, marker="o", label="Validation Error")
    plt.axhline(baseline_error, linestyle="--", linewidth=2, label=f"ZeroR Baseline = {baseline_error:.4f}")
    plt.xlabel("Training Set Size (increasing)")
    plt.ylabel("Log Loss")
    plt.title("Learning Curve\n(does more data help?)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n===== LEARNING CURVE =====")
    print(f"Baseline Error      : {baseline_error:.4f}")
    print(f"Final Validation Loss : {val_errors[-1]:.4f}")


def train_model():

    # ----------------------------------
    # Load Data
    # ----------------------------------
    df = fetch_training_data()
    df.columns = df.columns.str.lower()
    df = apply_feature_engineering(df)

    (
        X_scaled,
        y,
        scaler,
        encoders,
        feature_names,
        _
    ) = preprocess_data(df)

    raw_rule_features = extract_rule_baseline_features(df)
    assert len(raw_rule_features) == len(y), (
        "Row count mismatch between raw rule-baseline features and target — "
        "preprocess_data() may be dropping/reordering rows."
    )

    # ----------------------------------
    # Split: 60 Train / 20 Val / 20 Test
    # Raw rule-baseline features are split in lockstep so rows stay aligned
    # ----------------------------------
    (
        X_temp, X_test,
        y_temp, y_test,
        raw_temp, raw_test
    ) = train_test_split(
        X_scaled, y, raw_rule_features,
        test_size=0.20, random_state=42, stratify=y
    )

    (
        X_train, X_val,
        y_train, y_val,
        raw_train, raw_val
    ) = train_test_split(
        X_temp, y_temp, raw_temp,
        test_size=0.25, random_state=42, stratify=y_temp
    )

    # ----------------------------------
    # Grid Search
    # ----------------------------------
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    print("\n===== BEST PARAMETERS =====")
    print(grid.best_params_)

    # ----------------------------------
    # Diagnostics — WHY this C was chosen
    # (uses the real train/val split, no test data touched)
    # ----------------------------------
    plot_bias_variance_vs_regularization(
        X_train, y_train, X_val, y_val, C_values=param_grid["C"]
    )

    plot_learning_curve(X_train, y_train, X_val, y_val)

    # ----------------------------------
    # Threshold Tuning
    # ----------------------------------
    val_prob = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_precision = 0
    best_f1 = 0

    MIN_RECALL = 0.30
    highest_precision_seen = 0
    highest_precision_threshold = 0.50

    for threshold in np.arange(0.50, 0.99, 0.01):
        preds = (val_prob >= threshold).astype(int)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        if precision >= 0.90 and f1 > best_f1:
            best_threshold = threshold
            best_precision = precision
            best_f1 = f1

        if precision > highest_precision_seen and recall >= MIN_RECALL:
            highest_precision_seen = precision
            highest_precision_threshold = threshold

    if best_f1 == 0:
        best_threshold = highest_precision_threshold
        best_precision = highest_precision_seen
        val_preds = (val_prob >= best_threshold).astype(int)
        best_f1 = f1_score(y_val, val_preds, zero_division=0)
        print("\n[!] WARNING: No threshold reached 90% precision on validation data.")
        print(f"    Using best precision with recall>={MIN_RECALL}: {best_precision:.4f}")

    print("\n===== THRESHOLD TUNING =====")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Precision : {best_precision:.4f}")
    print(f"F1 Score  : {best_f1:.4f}")

    # ----------------------------------
    # Final Test Evaluation
    # ----------------------------------
    test_prob = model.predict_proba(X_test)[:, 1]
    test_preds = (test_prob >= best_threshold).astype(int)

    test_accuracy  = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall    = recall_score(y_test, test_preds, zero_division=0)
    test_f1        = f1_score(y_test, test_preds, zero_division=0)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy  : {test_accuracy:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")

    # ----------------------------------
    # Save Artifacts
    # ----------------------------------
    BASE_DIR = Path()
    artifacts_path = BASE_DIR / "app" / "artifacts"
    artifacts_path.mkdir(exist_ok=True)

    joblib.dump(model, artifacts_path / "model.pkl")
    joblib.dump(scaler, artifacts_path / "scaler.pkl")
    joblib.dump(encoders, artifacts_path / "encoders.pkl")
    joblib.dump(feature_names, artifacts_path / "feature_names.pkl")
    joblib.dump(best_threshold, artifacts_path / "threshold.pkl")

    joblib.dump(X_train, artifacts_path / "X_train.pkl")
    joblib.dump(y_train, artifacts_path / "y_train.pkl")
    joblib.dump(X_val, artifacts_path / "X_cv.pkl")
    joblib.dump(y_val, artifacts_path / "y_cv.pkl")
    joblib.dump(X_test, artifacts_path / "X_test.pkl")
    joblib.dump(y_test, artifacts_path / "y_test.pkl")
    # Raw (untransformed) features, for the rule-based baseline
    joblib.dump(raw_train, artifacts_path / "raw_train.pkl")
    joblib.dump(raw_val, artifacts_path / "raw_cv.pkl")
    joblib.dump(raw_test, artifacts_path / "raw_test.pkl")

    return {
        "best_params": grid.best_params_,
        "threshold": best_threshold,
        "accuracy": test_accuracy,
        "precision": test_precision,
        "recall": test_recall,
        "f1_score": test_f1
    }


if __name__ == "__main__":
    results = train_model()
    print(results)
evaluate_model()