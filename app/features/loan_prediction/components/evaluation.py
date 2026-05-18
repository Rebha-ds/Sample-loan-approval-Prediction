import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss
)
from app.features.loan_prediction.utils.preprocessing import apply_feature_engineering
from app.features.loan_prediction.components.data_ingestion import fetch_training_data
from app.features.loan_prediction.utils.preprocessing import preprocess_data


# -----------------------------------------------------------------------
# Cost Function
# -----------------------------------------------------------------------
def compute_cost_logistic(X, y, w, b):
    """
    Computes logistic regression cost function.
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = 1 / (1 + np.exp(-z_i))
        cost += -y[i] * np.log(f_wb_i + 1e-15) - (1 - y[i]) * np.log(1 - f_wb_i + 1e-15)
    cost = cost / m
    return cost
    


# -----------------------------------------------------------------------
# Cost Curve Plot (b-sweep and w-sweep, combined)
# -----------------------------------------------------------------------
def plot_cost_function(X, y):
    """
    Plots cost vs b (w fixed) and cost vs w-scale (b fixed) on the same graph.
    For the w-sweep a scalar multiplier alpha scales a unit direction vector so
    the x-axis represents a uniform weight scale.
    """
    n_features = X.shape[1]
    base_w  = np.ones(n_features)
    fixed_b = 0.0

    b_values     = np.linspace(-6, 0, 100)
    costs_b      = [compute_cost_logistic(X, y, base_w, b) for b in b_values]

    alpha_values = np.linspace(-3, 3, 100)
    costs_w      = [compute_cost_logistic(X, y, a * base_w, fixed_b) for a in alpha_values]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Logistic Regression Cost Function Analysis", fontsize=14, fontweight="bold")

    # Panel 1 - Cost vs b
    min_b_idx = np.argmin(costs_b)
    axes[0].plot(b_values, costs_b, color="steelblue", linewidth=2)
    axes[0].axvline(b_values[min_b_idx], color="tomato", linestyle="--", linewidth=1.2,
                    label=f"Min at b={b_values[min_b_idx]:.2f}")
    axes[0].set_xlabel("b (bias)", fontsize=11)
    axes[0].set_ylabel("Cost", fontsize=11)
    axes[0].set_title("Cost vs b\n(w = ones, fixed)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2 - Cost vs w scale
    min_w_idx = np.argmin(costs_w)
    axes[1].plot(alpha_values, costs_w, color="darkorange", linewidth=2)
    axes[1].axvline(alpha_values[min_w_idx], color="tomato", linestyle="--", linewidth=1.2,
                    label=f"Min at alpha={alpha_values[min_w_idx]:.2f}")
    axes[1].set_xlabel("w scale (alpha, where w = alpha * ones)", fontsize=11)
    axes[1].set_ylabel("Cost", fontsize=11)
    axes[1].set_title("Cost vs w scale\n(b = 0, fixed)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Panel 3 - Both overlaid (x normalised to [0,1])
    b_norm = (b_values - b_values.min()) / (b_values.max() - b_values.min())
    a_norm = (alpha_values - alpha_values.min()) / (alpha_values.max() - alpha_values.min())
    axes[2].plot(b_norm, costs_b, color="steelblue",  linewidth=2,
                 label="Cost vs b  (w=ones, b swept)")
    axes[2].plot(a_norm, costs_w, color="darkorange", linewidth=2, linestyle="--",
                 label="Cost vs w  (b=0, alpha swept)")
    axes[2].set_xlabel("Normalised parameter range [0 to 1]", fontsize=11)
    axes[2].set_ylabel("Cost", fontsize=11)
    axes[2].set_title("Cost vs b  &  Cost vs w\n(overlaid, x-axis normalised)", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("\n===== COST FUNCTION SUMMARY =====")
    print(f"Cost vs b  -> min cost = {min(costs_b):.4f}  at b = {b_values[min_b_idx]:.4f}")
    print(f"Cost vs w  -> min cost = {min(costs_w):.4f}  at alpha = {alpha_values[min_w_idx]:.4f}")


# -----------------------------------------------------------------------
# Bias / Variance helpers
# -----------------------------------------------------------------------
def _log_loss_score(model, X, y):
    """Return mean log-loss using the model's predicted probabilities."""
    proba = model.predict_proba(X)
    return log_loss(y, proba)


def plot_bias_variance_vs_regularization(X, y, baseline=None):
    """
    Trains logistic regression models for a range of regularisation values C
    (C = 1/lambda).  Plots training log-loss and CV log-loss vs lambda,
    mirroring the lab's 'lambda vs. train and CV MSEs' chart.

    A small C  = strong regularisation -> high bias risk.
    A large C  = weak  regularisation -> high variance risk.
    """
    X_train, X_cv, y_train, y_cv = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # C values large -> small  (mirrors lambda small -> large in the lab)
    C_values = [100, 50, 10, 5, 1, 0.5, 0.2, 0.1, 0.05, 0.01]
    train_errors, cv_errors = [], []

    for C in C_values:
        m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        m.fit(X_train, y_train)
        train_errors.append(_log_loss_score(m, X_train, y_train))
        cv_errors.append(_log_loss_score(m, X_cv, y_cv))

    lambda_labels = [f"{1/C:.2g}" for C in C_values]   # lambda = 1/C for x-axis

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_labels, train_errors, "ro-", linewidth=2, markersize=7, label="training log-loss")
    plt.plot(lambda_labels, cv_errors,    "bo-", linewidth=2, markersize=7, label="CV log-loss")
    if baseline is not None:
        plt.axhline(baseline, color="cyan", linestyle="--", linewidth=1.5, label="baseline")
    plt.xlabel("lambda  (regularisation strength = 1/C)", fontsize=11)
    plt.ylabel("Log-loss", fontsize=11)
    plt.title("lambda vs. Training and CV Log-Loss\n(Bias/Variance diagnosis)", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    best_idx   = int(np.argmin(cv_errors))
    best_C     = C_values[best_idx]
    best_train = train_errors[best_idx]
    best_cv    = cv_errors[best_idx]
    gap        = best_cv - best_train

    print("\n===== BIAS / VARIANCE DIAGNOSIS (regularisation sweep) =====")
    print(f"Best lambda = {1/best_C:.4g}  (C = {best_C})")
    print(f"  Training log-loss : {best_train:.4f}")
    print(f"  CV      log-loss  : {best_cv:.4f}")
    print(f"  Train-CV gap      : {gap:.4f}")

    if baseline is not None and best_train > baseline * 1.5:
        print("  [!] HIGH BIAS   - training error is well above baseline.")
        print("      -> Try: more features, lower lambda, more complex model.")
    elif gap > 0.10:
        print("  [!] HIGH VARIANCE - large gap between training and CV error.")
        print("      -> Try: increase lambda, fewer features, more training data.")
    else:
        print("  [OK] Model appears WELL-BALANCED at this lambda.")

    return C_values, train_errors, cv_errors


def plot_bias_variance_vs_training_size(X, y, C=1.0, baseline=None):
    """
    Trains the model on progressively larger subsets of the training data
    and plots learning curves (training log-loss and CV log-loss vs dataset
    size), mirroring the lab's 'number of examples vs. train and CV MSEs'.
    """
    X_train_full, X_cv, y_train_full, y_cv = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    m_total   = X_train_full.shape[0]
    fractions = np.linspace(0.10, 1.0, 10)
    sizes     = [max(int(f * m_total), 2) for f in fractions]

    train_errors, cv_errors = [], []

    for size in sizes:
        X_sub, y_sub = X_train_full[:size], y_train_full[:size]
        if len(np.unique(y_sub)) < 2:
            train_errors.append(np.nan)
            cv_errors.append(np.nan)
            continue
        m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        m.fit(X_sub, y_sub)
        train_errors.append(_log_loss_score(m, X_sub, y_sub))
        cv_errors.append(_log_loss_score(m, X_cv, y_cv))

    total_sizes = [s + len(X_cv) for s in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(total_sizes, train_errors, "ro-", linewidth=2, markersize=7, label="training log-loss")
    plt.plot(total_sizes, cv_errors,    "bo-", linewidth=2, markersize=7, label="CV log-loss")
    if baseline is not None:
        plt.axhline(baseline, color="cyan", linestyle="--", linewidth=1.5, label="baseline")
    plt.xlabel("total number of training and CV examples", fontsize=11)
    plt.ylabel("Log-loss", fontsize=11)
    plt.title(f"number of examples vs. Training and CV Log-Loss  (C={C})\nLearning curve / Bias-Variance diagnosis", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    final_train = train_errors[-1]
    final_cv    = cv_errors[-1]
    gap         = final_cv - final_train

    print("\n===== BIAS / VARIANCE DIAGNOSIS (learning curve) =====")
    print(f"  Final training log-loss : {final_train:.4f}")
    print(f"  Final CV      log-loss  : {final_cv:.4f}")
    print(f"  Train-CV gap            : {gap:.4f}")

    if baseline is not None and final_train > baseline * 1.5:
        print("  [!] HIGH BIAS   - adding more data will NOT help much.")
        print("      -> Try: more/better features, reduce regularisation.")
    elif gap > 0.10:
        print("  [!] HIGH VARIANCE - more training data may help close the gap.")
        print("      -> Try: more data, increase lambda, remove noisy features.")
    else:
        print("  [OK] Model appears WELL-BALANCED.")


# -----------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------
def evaluate_model():
    df = fetch_training_data()
    df = apply_feature_engineering(df)
    if df.empty:
        print("No labeled training data available for evaluation.")
        return
    
    print(df.columns)
    for feature in df.columns:
        if feature != "loan_status":
            sns.scatterplot(
                data=df,
                x=feature,
                y="loan_status",
                hue="loan_status",
                alpha=0.1
            )
        plt.show()
    X_scaled, y_true, s, e, f_n, model = preprocess_data(df)
    del s, e, f_n, model

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    ARTIFACT_DIR = "app/artifacts"
    MODEL_PATH   = os.path.join(ARTIFACT_DIR, "model.pkl")
    SCALER_PATH  = os.path.join(ARTIFACT_DIR, "scaler.pkl")
    ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoders.pkl")
    FEATURE_PATH = os.path.join(ARTIFACT_DIR, "feature_names.pkl")

    # ------------------------------------------------------------------
    # Load Artifacts
    # ------------------------------------------------------------------
    loaded_model  = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    encoders      = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    y_pred = loaded_model.predict(X_scaled)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)
    report    = classification_report(y_true, y_pred, zero_division=0)

    print("\n===== MODEL EVALUATION REPORT =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # ------------------------------------------------------------------
    # Cost function plots (b-sweep and w-sweep)
    # ------------------------------------------------------------------
    plot_cost_function(X_scaled, y_true)

    # ------------------------------------------------------------------
    # Bias / Variance diagnosis
    # Baseline = log-loss of the loaded model on the full dataset.
    # Override with a domain-specific float if needed, e.g. baseline=0.30
    # ------------------------------------------------------------------
    baseline_log_loss = _log_loss_score(loaded_model, X_scaled, y_true)
    print(f"\nBaseline log-loss (loaded model, full data): {baseline_log_loss:.4f}")

    # Plot 1: log-loss vs regularisation strength (lambda sweep)
    C_values, train_errs, cv_errs = plot_bias_variance_vs_regularization(
        X_scaled, y_true, baseline=baseline_log_loss
    )

    # Use best C from the sweep for the learning curve
    best_C = C_values[int(np.argmin(cv_errs))]

    # Plot 2: learning curve (log-loss vs training set size)
    plot_bias_variance_vs_training_size(
        X_scaled, y_true, C=best_C, baseline=baseline_log_loss
    )

    return {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "confusion_matrix": cm.tolist()
    }


# ------------------------------------------------------------------
# Run standalone
# ------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_model()