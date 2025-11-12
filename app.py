import streamlit as st
import numpy as np
import joblib
import os
import json

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
# Load / Save User Functions
# ================================
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    else:
        return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# Load users globally
users = load_users()

# ================================
# Session State Initialization
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ===============================
# Authentication Pages
# ===============================
def signup_page():
    st.title("🆕 Sign Up")
    email = st.text_input("Enter your Email ID")
    password = st.text_input("Create Password", type="password")

    if st.button("Sign Up"):
        users = load_users()  # Always reload current data
        if email in users:
            st.warning("⚠️ Account already exists! Please log in.")
        elif email == "" or password == "":
            st.error("❌ Please fill all fields.")
        else:
            users[email] = password
            save_users(users)
            st.success("✅ Account created successfully! Please log in now.")

def login_page():
    st.title("🔐 Login Page")
    email = st.text_input("Email ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()  # Get the latest stored users
        if email in users and users[email] == password:
            st.session_state.logged_in = True
            st.session_state.username = email
            st.success(f"✅ Welcome back, {email}!")
        else:
            st.error("❌ Invalid Email or Password!")

def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("👋 Logged out successfully!")

# ================================
# Helper: Check Parameter Levels
# ================================
def check_abnormalities(values):
    age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc = values
    messages = []

    if bp > 140:
        messages.append("⚠️ **High Blood Pressure** (>140 mmHg)")
    elif bp < 80:
        messages.append("⚠️ **Low Blood Pressure** (<80 mmHg)")

    if sg < 1.005:
        messages.append("⚠️ **Low Specific Gravity** — possible kidney filtration issue")

    if al > 1:
        messages.append("⚠️ **High Albumin** — protein in urine (proteinuria)")

    if su > 1:
        messages.append("⚠️ **High Urine Sugar** — may indicate diabetes")

    if bgr > 200:
        messages.append("⚠️ **High Blood Glucose** (>200 mg/dL)")
    elif bgr < 70:
        messages.append("⚠️ **Low Blood Glucose** (<70 mg/dL)")

    if sc > 1.5:
        messages.append("⚠️ **High Serum Creatinine** — kidney function reduced")

    if sod < 130:
        messages.append("⚠️ **Low Sodium** — electrolyte imbalance")

    if pot > 6:
        messages.append("⚠️ **High Potassium** — possible heart risk")
    elif pot < 3.5:
        messages.append("⚠️ **Low Potassium** — muscle weakness risk")

    if hemo < 11:
        messages.append("⚠️ **Low Hemoglobin** — anemia (common in CKD)")

    if wc > 11000:
        messages.append("⚠️ **High WBC Count** — possible infection or inflammation")
    elif wc < 4000:
        messages.append("⚠️ **Low WBC Count** — possible immune issue")

    if not messages:
        messages.append("✅ All test parameters are within normal range.")

    return messages

# ================================
# Main App Logic
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
    # CKD Prediction Section
    # ================================
    st.title("🩺 Chronic Kidney Disease Prediction System")
    st.subheader(f"Welcome, {st.session_state.username} 🎉")
    st.write("### Please enter patient health details below:")

    # ----------------------------
    # Input Fields
    # ----------------------------
    st.header("📋 Demographic Details")
    age = st.number_input("🧍 Age (years) [1 - 100]", 1, 100, 30)
    st.caption("✅ Any age (CKD risk increases after 40 years)")

    st.header(" Vital Signs")
    bp = st.number_input("Blood Pressure (mmHg) [50 - 180]", 50, 180, 120)
    st.caption("✅ Normal: 90–120 | ⚠️ High: >140")

    st.header("🧪 Urine Test Parameters")
    sg = st.number_input("Specific Gravity [1.00 - 1.03]", 1.00, 1.03, 1.02, step=0.001)
    al = st.number_input("Albumin (g/dL) [0 - 5]", 0, 5, 1)
    su = st.number_input("Sugar (mg/dL) [0 - 150]", 0, 150, 100)
    st.caption("✅ Normal: 0–150 mg/dL | ⚠️ High: >150 may indicate diabetes")

    st.header("🩸 Blood Test Parameters")
    bgr = st.number_input("Blood Glucose Random (mg/dL) [70 - 490]", 70, 490, 120)
    sc = st.number_input("Serum Creatinine (mg/dL) [0.4 - 15.0]", 0.4, 15.0, 1.2)
    sod = st.number_input("Sodium (mEq/L) [100 - 150]", 100, 150, 135)
    pot = st.number_input("Potassium (mEq/L) [2.5 - 7.0]", 2.5, 7.0, 4.5)
    hemo = st.number_input("Hemoglobin (g/dL) [3 - 17]", 3.0, 17.0, 13.5)
    pcv = st.number_input("Packed Cell Volume (%) [20 - 60]", 20, 60, 40)
    wc = st.number_input("White Blood Cell Count (cells/cumm) [2000 - 30000]", 2000, 30000, 8000)

    # ----------------------------
    # Prediction
    # ----------------------------
    features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
    features_scaled = scaler.transform(features)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("🔍 Predict CKD"):
        model = models[model_choice]
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            st.error("🚨 **Prediction: CKD Detected!**")
            st.subheader("🔎 Possible Causes:")
        else:
            st.success("✅ **Prediction: No CKD Detected!**")
            st.subheader("ℹ️ Health Observations:")

        reasons = check_abnormalities([age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc])
        for msg in reasons:
            st.write(msg)

    st.write("---")
    st.write("If you’re done, you can logout below:")
    logout_button()
