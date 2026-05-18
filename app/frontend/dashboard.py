import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"


# -----------------------------
# Session State
# -----------------------------
if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None


# -----------------------------
# Headers
# -----------------------------
headers = {}
if st.session_state.access_token:
    headers = {
        "Authorization": f"Bearer {st.session_state.access_token}"
    }


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Loan Prediction System")

if st.session_state.access_token:
    menu = st.sidebar.selectbox(
        "Navigation",
        [
            "Loan Prediction",
            "My Applications",
            "Logout"
        ]
    )
else:
    menu = st.sidebar.selectbox(
        "Navigation",
        [
            "Login",
            "Register"
        ]
    )


# -----------------------------
# Register Page
# -----------------------------
if menu == "Register":
    st.title("Create Account")

    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        payload = {
            "full_name": full_name,
            "email": email,
            "password": password
        }

        response = requests.post(
            f"{API_URL}/auth/register",
            json=payload
        )

        if response.status_code == 200:
            st.success("Account created successfully. Please login.")
        else:
            try:
                error_message = response.json().get("detail", "Registration failed")
            except:
                error_message = response.text or "Registration failed"

            st.error(error_message)
# -----------------------------
# Login Page
# -----------------------------
elif menu == "Login":
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        response = requests.post(
            f"{API_URL}/auth/login",
            json={
                "email": email,
                "password": password
            }
        )

        if response.status_code == 200:
            result = response.json()

            st.session_state.access_token = result["access_token"]
            st.session_state.user_email = result["user"]["email"]
            st.session_state.user_id = result["user"]["id"]

            st.success("Login successful")
            st.rerun()
        else:
            try:
                error_message = response.json().get("detail", "Invalid credentials")
            except:
                error_message = response.text or "Invalid credentials"
                st.error(error_message)
# -----------------------------
# Loan Prediction Page
# -----------------------------
elif menu == "Loan Prediction":
    st.title("Loan Eligibility Prediction")

    st.subheader("Enter Loan Details")

    gender = st.selectbox(
        "Gender",
        ["Rather not say", "Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Rather not say", "Yes", "No"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        step=1,
        value=0
    )

    education = st.selectbox(
        "Education",
        ["Rather not say", "Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Rather not say", "Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0
    )

    loan_amount = st.number_input(
        "Loan Amount (in dollars)",
        min_value=0.0
    )

    loan_amount_term = st.number_input(
        "Loan Amount Term (in months)",
        min_value=12
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    if st.button("Predict Loan Status"):
        payload = {
            "user_id": st.session_state.user_id,
            "gender": gender or None,
            "married": married or None,
            "dependents": dependents,
            "education": education or None,
            "self_employed": self_employed or None,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_amount_term": loan_amount_term,
            "credit_history": credit_history,
            "property_area": property_area
        }

        response = requests.post(
            f"{API_URL}/predict/",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Loan Status: {result['prediction']}")
            st.info(f"Loan ID: {result['loan_id']}")
            st.info(f"Application Number: {result['loan_number']}")
            st.info(f"Confidence Score: {result['confidence']}%")

        else:
            st.error(response.json().get("detail", "Prediction failed"))


# -----------------------------
# User Applications History
# -----------------------------
elif menu == "My Applications":
    st.title("My Loan Applications")

    response = requests.get(
        f"{API_URL}/predict/history/{st.session_state.user_id}",
        headers=headers
    )

    if response.status_code == 200:
        records = response.json()

        if records:
            df = pd.DataFrame(records)
            st.dataframe(df)
        else:
            st.info("No applications found.")
    else:
        st.error("Failed to fetch history")


# -----------------------------
# Logout
# -----------------------------
elif menu == "Logout":
    st.session_state.access_token = None
    st.session_state.user_email = None
    st.session_state.user_id = None

    st.success("Logged out successfully")
    st.rerun()
