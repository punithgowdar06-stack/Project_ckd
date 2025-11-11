import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ================================
# Load Models
# ================================
scaler = joblib.load("scaler.pkl")
models = {
    "Logistic Regression": joblib.load("logistic.pkl"),
    "Random Forest": joblib.load("randomforest.pkl"),
    "SVM": joblib.load("svm.pkl"),
    "KNN": joblib.load("knn.pkl"),
    "Naive Bayes": joblib.load("naivebayes.pkl")
}

# ================================
# Session State Initialization
# ================================
if "users" not in st.session_state:
    st.session_state.users = {}  # {username: password}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ================================
# Helper Functions
# ================================
def signup_page():
    st.title("🆕 Sign Up")
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    if st.button("Sign Up"):
        if username in st.session_state.users:
            st.error("❌ Username already exists! Please choose another.")
        elif username == "" or password == "":
            st.warning("⚠️ Please fill all fields.")
        else:
            st.session_state.users[username] = password
            st.success("✅ Account created successfully! Please go to Login page.")

def login_page():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful!")
        else:
            st.error("❌ Wrong username or password!")

def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("👋 Logged out successfully!")

# ================================
# Main Application Logic
# ================================
st.sidebar.title("Navigation")

if not st.session_state.logged_in:
    page = st.sidebar.radio("Go to", ["Login", "Sign Up"])

    if page == "Login":
        login_page()
    else:
        signup_page()

else:
    # ================================
    # Logged-In Home (CKD Prediction)
    # ================================
    st.title("🩺 Chronic Kidney Disease Prediction")
    st.subheader(f"Welcome, {st.session_state.username} 🎉")

    st.write("### Enter Patient Data Below")

    # Input form
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

    features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
    features = scaler.transform(features)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]
        prediction = model.predict(features)[0]
        result = "CKD Detected" if prediction == 1 else "No CKD"
        st.success(f"Prediction: {result}")

    st.write("---")
    st.write("If you’re done, you can logout below:")
    logout_button()



