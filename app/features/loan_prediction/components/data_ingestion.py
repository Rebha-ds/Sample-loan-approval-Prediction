import pandas as pd
from sqlalchemy import create_engine
from app.config.database import DATABASE_URL

engine = create_engine(DATABASE_URL)

def fetch_training_data():
    query = "SELECT * FROM loan_predict_train"
    print("data fetched from db")
    return pd.read_sql(query, engine)