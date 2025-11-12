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
            return json.load(f)
    else:
        return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# Load users on start
users = load_users()
# ================================
# Session State Initialization
# ================================
if "users" not in st.session_state:
    st.session_state.users = {}  # {email: password}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ================================
# Helper Functions
# ================================
def signup_page():
    st.title("🆕 Sign Up")
    email = st.text_input("Enter your Email ID")
    password = st.text_input("Create Password", type="password")

    if st.button("Sign Up"):
        if email in st.session_state.users:
            st.warning("⚠️ You already have an account. Please log in directly.")
        elif email == "" or password == "":
            st.error("❌ Please fill all fields.")
        else:
            st.session_state.users[email] = password
            st.success("✅ Account created successfully! Please log in now.")

def login_page():
    st.title("🔐 Login Page")
    email = st.text_input("Email ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in st.session_state.users and st.session_state.users[email] == password:
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
    # Input Fields with Hints
    # ----------------------------
    st.header("📋 Demographic Details")
    age = st.number_input("🧍 Age (years) [Range: 1 - 100]", min_value=1, max_value=100, value=30)
    st.caption("✅ Any age (CKD risk increases after 40 years)")

    st.header(" Vital Signs")
    bp = st.number_input("❤️ Blood Pressure (mmHg) [Range: 50 - 180]", min_value=50, max_value=180, value=120)
    st.caption("✅ Normal: 90–120 | ⚠️ High: >140")

    st.header("🧪 Urine Test Parameters")
    sg = st.number_input("⚗️ Specific Gravity [Range: 1.00 - 1.03]", min_value=1.00, max_value=1.03, value=1.02, step=0.001)
    st.caption("✅ Normal: 1.005–1.025")
    al = st.number_input("💧 Albumin (g/dL) [Range: 0 - 5]", min_value=0, max_value=5, value=1)
    st.caption("✅ Normal: 0 | ⚠️ High: >1 indicates protein in urine")
    su = st.number_input("🍬 Sugar (Urine Glucose Level) [Range: 0 - 5]", min_value=0, max_value=5, value=0)
    st.caption("✅ Normal: 0 | ⚠️ High: >1 may indicate diabetes")

    st.header("🩸 Blood Test Parameters")
    bgr = st.number_input("🩸 Blood Glucose Random (mg/dL) [Range: 70 - 490]", min_value=70, max_value=490, value=120)
    st.caption("✅ Normal: 70–140 | ⚠️ High: >200 may indicate diabetes")
    sc = st.number_input("🧬 Serum Creatinine (mg/dL) [Range: 0.4 - 15.0]", min_value=0.4, max_value=15.0, value=1.2)
    st.caption("✅ Normal: 0.6–1.2 | ⚠️ High: >1.5 may indicate kidney damage")
    sod = st.number_input("🧂 Sodium (mEq/L) [Range: 100 - 150]", min_value=100, max_value=150, value=135)
    st.caption("✅ Normal: 135–145")
    pot = st.number_input("🥔 Potassium (mEq/L) [Range: 2.5 - 7.0]", min_value=2.5, max_value=7.0, value=4.5)
    st.caption("✅ Normal: 3.5–5.5")
    hemo = st.number_input("🫀 Hemoglobin (g/dL) [Range: 3 - 17]", min_value=3.0, max_value=17.0, value=13.5)
    st.caption("✅ Normal: 12–17")
    pcv = st.number_input("🧫 Packed Cell Volume (%) [Range: 20 - 60]", min_value=20, max_value=60, value=40)
    st.caption("✅ Normal: 35–50")
    wc = st.number_input("🧪 White Blood Cell Count (cells/cumm) [Range: 2000 - 30000]", min_value=2000, max_value=30000, value=8000)
    st.caption("✅ Normal: 4000–11000")

    # ----------------------------
    # Prediction Section
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

        # Show explanation
        reasons = check_abnormalities([age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc])
        for msg in reasons:
            st.write(msg)

    st.write("---")
    st.write("If you’re done, you can logout below:")
    logout_button()
