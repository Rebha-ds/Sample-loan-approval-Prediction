# Loan Prediction System using Machine Learning

## Overview

This project is a full-stack Machine Learning-based Loan Prediction System designed to predict whether a loan application will be approved or rejected based on applicant details and financial information.

The system includes:

* Machine Learning model training and evaluation
* FastAPI backend APIs
* Streamlit frontend dashboard
* MySQL database integration
* JWT Authentication
* Automated model retraining
* Model evaluation and visualization

The application follows a modular production-style architecture with separate components for:

* preprocessing
* model training
* evaluation
* prediction
* authentication
* database management

---

# Features

## Authentication System

* User Registration
* User Login
* JWT Token Authentication
* Protected API Routes

---

## Loan Prediction

Predicts loan approval using:

* Logistic Regression
* Feature Engineering
* One-Hot Encoding
* Standard Scaling

Input features include:

* Gender
* Marital Status
* Dependents
* Education
* Self Employment
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

---

## Automated Loan ID Generation

Each prediction is automatically assigned:

* `loan_number`
* `loan_id`

Example:

| loan_number | loan_id  |
| ----------- | -------- |
| 1           | LP000001 |
| 25          | LP000025 |
| 105         | LP000105 |
| 1200        | LP001200 |

---

## Model Evaluation

Includes:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Cost Function Analysis
* Bias vs Variance Analysis
* Learning Curves
* ZeroR Baseline Comparison

---

# Tech Stack

## Backend

* Python
* FastAPI
* SQLAlchemy
* Pydantic
* Uvicorn
* JWT Authentication

---

## Frontend

* Streamlit

---

## Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Database

* MySQL

---

## Deployment

* Render (Backend)
* Streamlit Cloud (Frontend)
* Railway / MySQL Cloud

---

# Project Structure

```bash
app/
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_names.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
│
├── config/
│   └── database.py
│   └── security.py
│   └── settings.py
│
├── features/
│   ├── authentication/
│   └── loan_prediction/
│
├── frontend/
│   ├── dashboard.py
├── main.py
│
env.example
requirements.txt
README.md
```



# Installation

## Clone Repository

```bash
git clone <repository_url>

cd loan_prediction
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
DATABASE_URL=mysql+pymysql://username:password@host/database

SECRET_KEY=your_secret_key

ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

# Run Backend

```bash
uvicorn app.main:app --reload
```

Backend URL:

```bash
http://127.0.0.1:8000
```

---

# Run Frontend

```bash
streamlit run dashboard.py
```

---

# Model Training

```bash
python -m app.features.loan_prediction.components.model_trainer
```

---

# Model Evaluation

```bash
python -m app.features.loan_prediction.components.evaluation
```

---

# API Endpoints

## Authentication

### Register

```http
POST /auth/register
```

### Login

```http
POST /auth/login
```

---

## Loan Prediction

### Predict Loan

```http
POST /predict
```

---

# Machine Learning Workflow

## Training Pipeline

1. Fetch training data
2. Apply feature engineering
3. Preprocess data
4. One-hot encode categorical columns
5. Scale numeric features
6. Train Logistic Regression model
7. Save artifacts

---

## Evaluation Pipeline

1. Load trained model
2. Load test dataset
3. Generate predictions
4. Compute evaluation metrics
5. Plot graphs and diagnostics

---

# Evaluation Graphs

The system generates:

* Confusion Matrix
* Cost vs Bias
* Cost vs Weight Scale
* Bias vs Variance
* Learning Curve
* Baseline Error Comparison

---

# Future Improvements

* XGBoost / Random Forest support
* SHAP explainability
* Docker containerization
* CI/CD pipelines
* Admin analytics dashboard
* Role-based authorization
* Model monitoring
* Drift detection

---

# Author

Developed as a full-stack Machine Learning project integrating:

* Machine Learning
* Backend Development
* Frontend Development
* Database Management
* Model Evaluation
* Deployment Pipelines

