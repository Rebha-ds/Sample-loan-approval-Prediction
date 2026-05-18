from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.settings import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME
)

# -----------------------------
# Database URL
# -----------------------------
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -----------------------------
# SQLAlchemy Engine
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

# -----------------------------
# Session Factory
# -----------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# -----------------------------
# Base Class
# -----------------------------
Base = declarative_base()


# -----------------------------
# Import ALL models here
# VERY IMPORTANT:
# This ensures SQLAlchemy registers all tables
# before create_all() runs
# -----------------------------
from app.features.authentication.models.user_model import User
from app.features.loan_prediction.models.loan_model import LoanPredictTrain


# -----------------------------
# Create Tables
# -----------------------------
Base.metadata.create_all(bind=engine)


# -----------------------------
# Dependency for FastAPI
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
