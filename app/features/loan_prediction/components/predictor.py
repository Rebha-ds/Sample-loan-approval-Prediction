import os
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from app.features.loan_prediction.models.loan_model import LoanPredictTrain

ARTIFACT_DIR = "app/artifacts"
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH  = os.path.join(ARTIFACT_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoders.pkl")
FEATURE_PATH = os.path.join(ARTIFACT_DIR, "feature_names.pkl")

model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
encoders  = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_PATH)


def preprocess_payload(payload):
    data = {
        "gender":            payload.gender or "Unknown",
        "married":           payload.married or "Unknown",
        "dependents":        payload.dependents if payload.dependents is not None else 0,
        "education":         payload.education or "Unknown",
        "self_employed":     payload.self_employed or "Unknown",
        "applicantincome":   payload.applicant_income if payload.applicant_income is not None else 0,
        "coapplicantincome": payload.coapplicant_income if payload.coapplicant_income is not None else 0,
        "loanamount":        payload.loan_amount if payload.loan_amount is not None else 0,
        "loan_amount_term":  payload.loan_amount_term,
        "credit_history":    payload.credit_history,
        "property_area":     payload.property_area or "Unknown",
    }

    df = pd.DataFrame([data])
    df.columns = df.columns.str.lower()

    fill_defaults = {
        "gender": "Unknown", "married": "Unknown", "dependents": 0,
        "education": "Unknown", "self_employed": "Unknown",
        "applicantincome": 0, "coapplicantincome": 0,
        "loanamount": 0, "property_area": "Unknown",
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    numeric_cols = [
        "dependents", "applicantincome", "coapplicantincome",
        "loanamount", "loan_amount_term", "credit_history",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return encoders.transform(df)


def predict_loan(payload, db: Session):
    processed_data = preprocess_payload(payload)

    prediction  = model.predict(processed_data)[0]
    confidence  = max(model.predict_proba(processed_data)[0]) * 100
    prediction_result = "Approved" if prediction == 1 else "Rejected"

    return prediction_result, round(confidence, 2)          # no retrained flag


def save_prediction(db, user_id, payload, prediction, confidence):
    new_record = LoanPredictTrain(
        user_id=user_id,
        gender=payload.gender,
        married=payload.married,
        dependents=payload.dependents,
        education=payload.education,
        self_employed=payload.self_employed,
        applicantincome=payload.applicant_income,
        coapplicantincome=payload.coapplicant_income,
        loanamount=payload.loan_amount,
        loan_amount_term=payload.loan_amount_term,
        credit_history=payload.credit_history,
        property_area=payload.property_area,
        prediction=prediction,
        confidence=confidence,
    )
    
    try:
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
    except Exception as e:
        db.rollback()
        print(e)
    return new_record