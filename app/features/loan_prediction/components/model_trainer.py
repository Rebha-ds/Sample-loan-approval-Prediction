import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from app.features.loan_prediction.components.data_ingestion import fetch_training_data
from app.features.loan_prediction.utils.preprocessing import preprocess_data
from app.features.loan_prediction.components.evaluation import evaluate_model
from app.features.loan_prediction.utils.preprocessing import apply_feature_engineering


def train_model():
    df = fetch_training_data()
    df.columns=df.columns.str.lower()
    
    df = apply_feature_engineering(df)    
    X_scaled, y,  scaler, encoders, feature_names,model  = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    BASE_DIR = Path()   # adjust as needed
    artifacts_path = BASE_DIR / "app" / "artifacts"

    joblib.dump(model, artifacts_path / "model.pkl")
    joblib.dump(scaler, artifacts_path / "scaler.pkl")
    joblib.dump(encoders, artifacts_path / "encoders.pkl")
    joblib.dump(feature_names, artifacts_path/"feature_names.pkl")

    return acc

train_model()
evaluate_model()