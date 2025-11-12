import streamlit as st
import numpy as np
import joblib
import pickle
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
# File to store user data
# ================================
USER_FILE = "users.json"

# ================================


# ================================
# Helper: Save & Load Users
# ================================
def save_users(users):
    with open("users.pkl", "wb") as f:
        pickle.dump(users, f)

def load_users():
    if os.path.exists("users.pkl"):
        with open("users.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# ================================
# Session Initialization
# ================================
if "users" not in st.session_state:
    st.session_state.users = load_users()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ================================
# Authentication Pages
# ================================
def signup_page():
    st.title("🆕 Sign Up")
    email = st.text_input("Enter Email ID (Username)")
    password = st.text_input("Create Password", type="password")

    if st.button("Sign Up"):
        if email in st.session_state.users:
            st.warning("⚠️ Account already exists. Please go to login.")
        elif email == "" or password == "":
            st.warning("⚠️ Please fill all fields.")
        else:
            st.session_state.users[email] = password
            save_users(st.session_state.users)
            st.success("✅ Account created successfully! Please login.")

def login_page():
    st.title("🔐 Login Page")
    email = st.text_input("Email ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in st.session_state.users and st.session_state.users[email] == password:
            st.session_state.logged_in = True
            st.session_state.username = email
            st.success("✅ Login successful!")
        else:
            st.error("❌ Incorrect Email or Password!")

def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("👋 Logged out successfully!")

# ================================
# Check Abnormalities Function
# ================================
def check_abnormalities(values):
    age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc = values
    messages = []

    if bp > 140:
        messages.append("⚠️ High Blood Pressure (>140 mmHg)")
    elif bp < 80:
        messages.append("⚠️ Low Blood Pressure (<80 mmHg)")
    if sg < 1.005:
        messages.append("⚠️ Low Specific Gravity — possible kidney filtration issue")
    if al > 1:
        messages.append("⚠️ High Albumin — protein in urine")
    if su > 150:
        messages.append("⚠️ High Blood Sugar (>150 mg/dL) — possible diabetes")
    elif su < 70:
        messages.append("⚠️ Low Blood Sugar (<70 mg/dL) — possible hypoglycemia")
    if bgr > 200:
        messages.append("⚠️ High Blood Glucose (>200 mg/dL)")
    elif bgr < 70:
        messages.append("⚠️ Low Blood Glucose (<70 mg/dL)")
    if sc > 1.5:
        messages.append("⚠️ High Serum Creatinine — kidney function reduced")
    if sod < 130:
        messages.append("⚠️ Low Sodium — electrolyte imbalance")
    if pot > 6:
        messages.append("⚠️ High Potassium — heart risk")
    elif pot < 3.5:
        messages.append("⚠️ Low Potassium — muscle weakness risk")
    if hemo < 11:
        messages.append("⚠️ Low Hemoglobin — anemia (common in CKD)")
    if wc > 11000:
        messages.append("⚠️ High WBC Count — infection/inflammation")
    elif wc < 4000:
        messages.append("⚠️ Low WBC Count — immune issue")
    if not messages:
        messages.append("✅ All test parameters are within normal range.")
    return messages

# ================================
# Main CKD Prediction Page
# ================================
def ckd_prediction_page():
    st.title("🩺 Chronic Kidney Disease Prediction System")
    st.subheader(f"Welcome, {st.session_state.username} 🎉")
    st.write("### Enter the following patient details below:")

    age = st.number_input("Age (years) [Range: 0 - 100]", 1, 100, 30)
    bp = st.number_input("Blood Pressure (mmHg) [Range: 50 - 300]", 50, 300, 120)
    sg = st.number_input("Specific Gravity [Range: 1.00 - 1.03]", 1.00, 1.03, 1.02, step=0.001)
    al = st.number_input("Albumin (g/dL) [Range: 0 - 5]", 0, 5, 1)
    su = st.number_input("Blood Sugar (mg/dL) [Range: 0 - 150]", 0, 150, 90)
    bgr = st.number_input("Blood Glucose Random (mg/dL) [Range: 50 - 500]", 50, 500, 120)
    sc = st.number_input("Serum Creatinine (mg/dL) [Range: 0.4 - 15]", 0.4, 15.0, 1.2)
    sod = st.number_input("Sodium (mEq/L) [Range: 100 - 200]", 100, 200, 135)
    pot = st.number_input("Potassium (mEq/L) [Range: 2.5 - 7]", 2.5, 7.0, 4.5)
    hemo = st.number_input("Hemoglobin (g/dL) [Range: 3 - 20]", 3.0, 20.0, 13.5)
    pcv = st.number_input("Packed Cell Volume (%) [Range: 20 - 60]", 20, 60, 40)
    wc = st.number_input("White Blood Cell Count (cells/cumm) [Range: 4000 - 20000]", 4000, 20000, 8000)

    features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
    features_scaled = scaler.transform(features)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("🔍 Predict CKD"):
        model = models[model_choice]
        prediction = model.predict(features_scaled)[0]
        abnormalities = check_abnormalities([age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc])

        if prediction == 1:
            st.error("🚨 CKD Detected!")
            st.write("### ⚠️ Possible Reasons:")
            for msg in abnormalities:
                st.write("- ", msg)
        else:
            st.success("✅ No CKD Detected")
            st.write("### ℹ️ Observations:")
            for msg in abnormalities:
                st.write("- ", msg)

    st.write("---")
    logout_button()

# ================================
# Navigation Logic
# ================================
st.sidebar.title("Navigation")

if not st.session_state.logged_in:
    page = st.sidebar.radio("Go to", ["Login", "Sign Up"])
    if page == "Login":
        login_page()
    else:
        signup_page()
else:
    ckd_prediction_page()
