from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.features.authentication.routes.auth_routes import router as auth_router
from app.features.loan_prediction.routes.prediction_routes import router as prediction_router

app = FastAPI(
    title="Loan Prediction System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],   # Streamlit dev URL
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Loan Prediction System API is running successfully"
    }


# Register feature routes
app.include_router(auth_router)
app.include_router(prediction_router)






