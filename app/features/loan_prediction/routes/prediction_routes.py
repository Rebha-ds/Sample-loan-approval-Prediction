from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.config.database import get_db
from app.features.loan_prediction.components.predictor import predict_loan, save_prediction

router = APIRouter(prefix="/predict", tags=["Loan Prediction"])


class LoanPredictionRequest(BaseModel):
    user_id: int
    gender:            Optional[str]   = None
    married:           Optional[str]   = None
    dependents:        Optional[int]   = None
    education:         Optional[str]   = None
    self_employed:     Optional[str]   = None
    applicant_income:  Optional[int]   = None
    coapplicant_income: Optional[int]  = None
    loan_amount:       Optional[float] = None
    loan_amount_term:  float
    credit_history:    float
    property_area:     str


class LoanPredictionResponse(BaseModel):
    loan_number: Optional[int]=None
    loan_id:     Optional[str]=None
    prediction:  str
    confidence:  float                  # model_retrained removed


@router.post("/", response_model=LoanPredictionResponse)
def create_prediction(payload: LoanPredictionRequest, db: Session = Depends(get_db)):
    try:
        prediction_result, confidence_score = predict_loan(payload, db)

        saved_record = save_prediction(
            db=db,
            user_id=payload.user_id,
            payload=payload,
            prediction=prediction_result,
            confidence=confidence_score,
        )

        return LoanPredictionResponse(
            loan_number=saved_record.loan_number,
            loan_id=saved_record.loan_id,
            prediction=prediction_result,
            confidence=confidence_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history/{user_id}")
def get_user_prediction_history(user_id: int, db: Session = Depends(get_db)):
    from app.features.loan_prediction.models.loan_model import LoanPredictTrain
    records = (
        db.query(LoanPredictTrain)
        .filter(LoanPredictTrain.user_id == user_id)
        .order_by(LoanPredictTrain.created_at.desc())
        .all()
    )
    return [
        {
            "loan_number": r.loan_number,
            "loan_id":     r.loan_id,
            "prediction":  r.prediction,
            "confidence":  r.confidence,
            "created_at":  r.created_at,
        }
        for r in records
    ]