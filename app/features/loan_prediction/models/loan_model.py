from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, text
from app.config.database import Base


class LoanPredictTrain(Base):
    __tablename__ = "loan_predict_train"

    loan_number = Column("loan_number", Integer, primary_key=True, index=True)
    user_id     = Column("user_id", Integer, ForeignKey("users.id"), nullable=False)

    loan_id       = Column("Loan_ID",       String(20), unique=True, nullable=True)
    gender        = Column("Gender",        String(10), nullable=True)
    married       = Column("Married",       String(10), nullable=True)
    dependents    = Column("Dependents",    Integer,    nullable=True)
    education     = Column("Education",     String(45), nullable=True)
    self_employed = Column("Self_Employed", String(20), nullable=True)

    applicantincome   = Column("ApplicantIncome",   Integer, nullable=True)
    coapplicantincome = Column("CoapplicantIncome", Integer, nullable=True)
    loanamount        = Column("LoanAmount",        Float,   nullable=True)

    loan_amount_term = Column("Loan_Amount_Term", Float,      nullable=False)
    credit_history   = Column("Credit_History",   Float,      nullable=False)
    property_area    = Column("Property_Area",    String(20), nullable=False)

    prediction = Column("prediction", String(20), nullable=False)
    confidence = Column("confidence", Float,      nullable=False)
    loan_status = Column("Loan_Status", String(10), nullable=True)

    created_at = Column(
        "created_at", TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )