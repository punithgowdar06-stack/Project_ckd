import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.title("🩺 Chronic Kidney Disease Prediction")

# --- Login Functionality ---
def check_password():
    """Returns `True` if the user entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input again.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# --- Main App Content (after login) ---
if check_password():
    st.success("Logged in successfully!")

    # Load models
    try:
        scaler = joblib.load("models/scaler.pkl")
        models = {
            "Logistic Regression": joblib.load("logistic.pkl"),
            "Random Forest": joblib.load("randomforest.pkl"),
            "SVM": joblib.load("svm.pkl"),
            "KNN": joblib.load("knn.pkl"),
            "Naive Bayes": joblib.load("naivebayes.pkl")
        }
    except FileNotFoundError:
        st.error("Error: Model files not found. Make sure the 'models' directory and its contents are in the correct location.")
        st.stop()


    # Input form
    st.subheader("Enter Patient Data")
    age = st.number_input("Age", 1, 100)
    bp = st.number_input("Blood Pressure", 50, 200)
    sg = st.number_input("Specific Gravity", 1.0, 1.025)
    al = st.number_input("Albumin", 0, 5)
    su = st.number_input("Sugar", 0, 5)
    bgr = st.number_input("Blood Glucose Random", 50, 500)
    sc = st.number_input("Serum Creatinine", 0.1, 15.0)
    sod = st.number_input("Sodium", 100, 200)
    pot = st.number_input("Potassium", 2.0, 10.0)
    hemo = st.number_input("Hemoglobin", 3.0, 17.0)
    pcv = st.number_input("Packed Cell Volume", 20, 60)
    wc = st.number_input("White Blood Cell Count", 2000, 30000)

    # features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc, rc]]) # Modified this line
    features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
    features = scaler.transform(features)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]
        prediction = model.predict(features)[0]
        result = "CKD Detected" if prediction == 1 else "No CKD"
        st.success(f"Prediction: {result}")

        # Display image based on prediction
        if prediction == 1:
            try:
                image = Image.open('kidney_failure.png') # Replace with your image path
                st.image(image, caption='Kidney Failure Detected', use_column_width=True)
            except FileNotFoundError:
                st.error("Kidney failure image not found. Make sure 'kidney_failure.png' is in the correct location.")
        else:
            try:
                image = Image.open('clear_kidney.png') # Replace with your image path
                st.image(image, caption='Kidney Looks Clear', use_column_width=True)
            except FileNotFoundError:
                 st.error("Clear kidney image not found. Make sure 'clear_kidney.png' is in the correct location.")

    # --- Logout Button ---
    if st.button("Logout"):
        st.session_state["password_correct"] = False
        st.experimental_return()
