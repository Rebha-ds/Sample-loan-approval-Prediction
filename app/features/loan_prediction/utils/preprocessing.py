# app/features/loan_prediction/utils/preprocessing.py

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LogisticRegression


def preprocess_data(df, model=None):
    """
    Preprocesses loan dataset for model training.

    Returns:
        X_scaled
        y
        scaler
        encoders
        feature_names
        model
    """

    # -----------------------------
    # Standardize columns
    # -----------------------------
    df.columns = df.columns.str.lower()

    # -----------------------------
    # Drop unnecessary columns
    # -----------------------------
    removable_cols = [
        "loan_id",
        "created_at",
        "loan_number",
        "user_id",
        "retrained_at",
        "prediction",
        "confidence"
    ]

    existing_cols = [
        col for col in removable_cols
        if col in df.columns
    ]

    df.drop(columns=existing_cols, inplace=True)

    # -----------------------------
    # Fill null values
    # -----------------------------
    fill_defaults = {
        "gender": "Unknown ",
        "married": "Unknown",
        "dependents": 0,
        "education": "Unknown",
        "self_employed": "Unknown",
        "applicantincome": 0,
        "coapplicantincome": 0,
        "loanamount": 0,
        "property_area": "Unknown"
    }

    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # -----------------------------
    # Numeric conversion
    # -----------------------------
    numeric_cols = [
        "dependents",
        "applicantincome",
        "coapplicantincome",
        "loanamount",
        "loan_amount_term",
        "credit_history"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # -----------------------------
    # Target
    # -----------------------------
    y = df["loan_status"].apply(
        lambda x: 1 if str(x).upper() in ["Y", "APPROVED", "1"] else 0
    )

    X = df.drop(columns=["loan_status"])

    # -----------------------------
    # Categorical + Numerical split
    # -----------------------------
    categorical_cols = [
        col for col in [
            "gender",
            "married",
            "education",
            "self_employed",
            "property_area"
        ]
        if col in X.columns
    ]

    numerical_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    # -----------------------------
    # Encoders
    # -----------------------------
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    scaler = StandardScaler()

    encoders = ColumnTransformer(
        transformers=[
            (
                "cat",
                onehot_encoder,
                categorical_cols
            ),
            (
                "num",
                scaler,
                numerical_cols
            )
        ]
    )

    # -----------------------------
    # Transform
    # -----------------------------
    X_scaled = encoders.fit_transform(X)

    # -----------------------------
    # Feature names
    # -----------------------------
    cat_feature_names = list(
        encoders.named_transformers_["cat"].get_feature_names_out(
            categorical_cols
        )
    )

    feature_names = X.columns.tolist()
    
    model = LogisticRegression()
    return (
        X_scaled,
        y,
        scaler,
        encoders,
        feature_names,
        model
    )

# preprocessing.py (or a shared utils file)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies consistent power transformations before scaling/encoding."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    df['credit_history']    = df['credit_history'].astype('Int64') ** 5
    df['applicantincome']   = df['applicantincome'].astype('int64') ** 4
    df['loanamount']        = pd.to_numeric(df['loanamount'],        errors='coerce').fillna(0).astype('int64') ** 5
    df['coapplicantincome'] = pd.to_numeric(df['coapplicantincome'], errors='coerce').fillna(0).astype('int64') ** 4
    df['loan_amount_term']  = pd.to_numeric(df['loan_amount_term'],  errors='coerce').fillna(0).astype('int64') ** 3
    df['property_area']     = pd.to_numeric(df['property_area'],     errors='coerce').fillna(0).astype('int64') ** 2
    return df