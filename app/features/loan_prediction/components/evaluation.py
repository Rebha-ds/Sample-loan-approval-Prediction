# app/features/loan_prediction/components/evaluation.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    precision_recall_curve,
    average_precision_score
)

from app.features.loan_prediction.utils.preprocessing import (
    rule_based_loan_baseline          # <-- added
)


# =========================================================
# Artifact Paths
# =========================================================
ARTIFACT_DIR = "app/artifacts"

MODEL_PATH      = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH     = os.path.join(ARTIFACT_DIR, "scaler.pkl")
ENCODER_PATH    = os.path.join(ARTIFACT_DIR, "encoders.pkl")
FEATURE_PATH    = os.path.join(ARTIFACT_DIR, "feature_names.pkl")
X_TRAIN_PATH    = os.path.join(ARTIFACT_DIR, "X_train.pkl")
X_CV_PATH       = os.path.join(ARTIFACT_DIR, "X_cv.pkl")
X_TEST_PATH     = os.path.join(ARTIFACT_DIR, "X_test.pkl")
Y_TRAIN_PATH    = os.path.join(ARTIFACT_DIR, "y_train.pkl")
Y_CV_PATH       = os.path.join(ARTIFACT_DIR, "y_cv.pkl")
Y_TEST_PATH     = os.path.join(ARTIFACT_DIR, "y_test.pkl")
THRESHOLD_PATH  = os.path.join(ARTIFACT_DIR, "threshold.pkl")
RAW_TEST_PATH   = os.path.join(ARTIFACT_DIR, "raw_test.pkl")       # <-- added


# =========================================================
# Cost Function
# =========================================================
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = 1 / (1 + np.exp(-z_i))
        cost += (
            -y.iloc[i] * np.log(f_wb_i + 1e-15)
            - (1 - y.iloc[i]) * np.log(1 - f_wb_i + 1e-15)
        )
    return cost / m


def plot_cost_function(X, y):
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X, y)
    baseline_error = log_loss(y, baseline_model.predict_proba(X))

    n_features = X.shape[1]
    base_w = np.ones(n_features)
    fixed_b = 0.0

    b_values = np.linspace(-6, 6, 100)
    costs_b = [compute_cost_logistic(X, y, base_w, b) for b in b_values]
    min_b_index = np.argmin(costs_b)
    best_b, best_b_cost = b_values[min_b_index], costs_b[min_b_index]

    alpha_values = np.linspace(-3, 3, 100)
    costs_w = [compute_cost_logistic(X, y, a * base_w, fixed_b) for a in alpha_values]
    min_alpha_index = np.argmin(costs_w)
    best_alpha, best_alpha_cost = alpha_values[min_alpha_index], costs_w[min_alpha_index]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(b_values, costs_b, linewidth=2, label="Cost Curve")
    axes[0].scatter(best_b, best_b_cost, s=100, zorder=5, label=f"Minimum Cost\nb={best_b:.2f}")
    axes[0].axvline(best_b, linestyle="--", linewidth=1.5)
    axes[0].axhline(baseline_error, linestyle=":", linewidth=2, label=f"ZeroR Baseline = {baseline_error:.4f}")
    axes[0].set_title("Cost vs Bias (b)")
    axes[0].set_xlabel("Bias (b)")
    axes[0].set_ylabel("Cost")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(alpha_values, costs_w, linewidth=2, label="Cost Curve")
    axes[1].scatter(best_alpha, best_alpha_cost, s=100, zorder=5, label=f"Minimum Cost\nα={best_alpha:.2f}")
    axes[1].axvline(best_alpha, linestyle="--", linewidth=1.5)
    axes[1].axhline(baseline_error, linestyle=":", linewidth=2, label=f"ZeroR Baseline = {baseline_error:.4f}")
    axes[1].set_title("Cost vs Weight Scale")
    axes[1].set_xlabel("Weight Scale (α)")
    axes[1].set_ylabel("Cost")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print("\n===== COST FUNCTION ANALYSIS =====")
    print(f"Best b value        : {best_b:.4f}")
    print(f"Minimum cost (b)    : {best_b_cost:.4f}")
    print(f"Best alpha value    : {best_alpha:.4f}")
    print(f"Minimum cost (w)    : {best_alpha_cost:.4f}")
    print(f"ZeroR baseline loss : {baseline_error:.4f}")


def plot_precision_recall_curve(y_true, y_proba, chosen_threshold=None):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2, label=f"PR Curve (AP={avg_precision:.4f})")

    if chosen_threshold is not None:
        idx = np.argmin(np.abs(thresholds - chosen_threshold))
        plt.scatter(
            recalls[idx], precisions[idx],
            color="red", s=100, zorder=5,
            label=f"Production threshold = {chosen_threshold:.2f}\n"
                  f"(P={precisions[idx]:.4f}, R={recalls[idx]:.4f})"
        )

    plt.axhline(0.90, linestyle="--", color="gray", linewidth=1, label="90% Precision target")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n===== PRECISION-RECALL CURVE =====")
    print(f"Average Precision (AP) : {avg_precision:.4f}")
    if chosen_threshold is not None:
        idx = np.argmin(np.abs(thresholds - chosen_threshold))
        print(f"At production threshold {chosen_threshold:.2f}: "
              f"Precision={precisions[idx]:.4f}, Recall={recalls[idx]:.4f}")


# =========================================================
# Accuracy Bar Chart — now includes the rule-based baseline
# =========================================================
def plot_accuracy_bar_chart(model, X_train, y_train, X_cv, y_cv, X_test, y_test,
                             threshold, raw_test):

    # --- ZeroR baseline ---
    zeror_model = DummyClassifier(strategy="most_frequent")
    zeror_model.fit(X_train, y_train)
    zeror_acc = accuracy_score(y_test, zeror_model.predict(X_test))

    # --- Rule-based baseline (60% income ratio + credit_history) ---
    rule_preds = rule_based_loan_baseline(raw_test)
    rule_acc       = accuracy_score(y_test, rule_preds)
    rule_precision = precision_score(y_test, rule_preds, zero_division=0)
    rule_recall    = recall_score(y_test, rule_preds, zero_division=0)
    rule_f1        = f1_score(y_test, rule_preds, zero_division=0)

    # --- Actual trained model, no refitting ---
    train_pred = (model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
    cv_pred    = (model.predict_proba(X_cv)[:, 1] >= threshold).astype(int)
    test_pred  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    train_acc = accuracy_score(y_train, train_pred)
    cv_acc    = accuracy_score(y_cv, cv_pred)
    test_acc  = accuracy_score(y_test, test_pred)

    categories = ["ZeroR Baseline", "Rule-Based Baseline", "Train", "CV", "Test"]
    accuracies = [zeror_acc, rule_acc, train_acc, cv_acc, test_acc]
    colors = ["gray", "darkorange", "tomato", "steelblue", "seagreen"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, accuracies, color=colors)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.4f}",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy: ZeroR vs Rule-Based vs Train vs CV vs Test\n(actual production model, no refitting)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

    print("\n===== ACCURACY BAR CHART =====")
    print(f"ZeroR Baseline Accuracy       : {zeror_acc:.4f}")
    print(f"Rule-Based Baseline Accuracy  : {rule_acc:.4f}")
    print(f"  (rule: precision={rule_precision:.4f}, recall={rule_recall:.4f}, f1={rule_f1:.4f})")
    print(f"Train Accuracy                : {train_acc:.4f}")
    print(f"CV Accuracy                   : {cv_acc:.4f}")
    print(f"Test Accuracy                 : {test_acc:.4f}")

    if test_acc <= rule_acc:
        print("[!] WARNING: Model does not outperform the rule-based baseline on test accuracy.")
    elif test_acc <= zeror_acc:
        print("[!] WARNING: Model does not outperform the ZeroR baseline on test accuracy.")
    else:
        print("[OK] Model outperforms both baselines on test accuracy.")

    gap = train_acc - cv_acc
    if gap > 0.10:
        print(f"[!] HIGH VARIANCE - large train/CV accuracy gap ({gap:.4f}).")


# =========================================================
# Main Evaluation
# =========================================================
def evaluate_model():

    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    encoders      = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    threshold     = joblib.load(THRESHOLD_PATH)

    X_train = joblib.load(X_TRAIN_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    X_cv    = joblib.load(X_CV_PATH)
    y_cv    = joblib.load(Y_CV_PATH)
    X_test  = joblib.load(X_TEST_PATH)
    y_test  = joblib.load(Y_TEST_PATH)
    raw_test = joblib.load(RAW_TEST_PATH)        # <-- added

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, zero_division=0)

    print("\n===== MODEL EVALUATION (Test Set) =====")
    print(f"Threshold used : {threshold:.2f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if precision < 0.90:
        print("[!] WARNING: Precision target of 0.90 was NOT met on the test set.")
    else:
        print("[OK] Precision target of 0.90 met.")

    print("\n===== CONFUSION MATRIX =====")
    print(cm)
    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

    plot_precision_recall_curve(y_test, y_proba, chosen_threshold=threshold)

    plot_cost_function(X_train, y_train)

    plot_accuracy_bar_chart(
        model, X_train, y_train, X_cv, y_cv, X_test, y_test, threshold, raw_test
    )

    return {
        "threshold":        threshold,
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    evaluate_model()