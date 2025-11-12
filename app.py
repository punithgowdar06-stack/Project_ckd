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
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ================================
# Sign Up Page
# ================================
def signup_page():
    st.title("🆕 Create Your Account")
    email = st.text_input("Enter your Email ID")
    password = st.text_input("Create Password", type="password")

    if st.button("Sign Up"):
        if email == "" or password == "":
            st.warning("⚠️ Please fill all fields.")
        elif email in users:
            st.warning("⚠️ Account already exists! Please login.")
        else:
            users[email] = password
            save_users(users)
            st.success("✅ Account created successfully! Please go to Login page.")

# ================================
# Login Page
# ================================
def login_page():
    st.title("🔐 Login Page")
    email = st.text_input("Email ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in users and users[email] == password:
            st.session_state.logged_in = True
            st.session_state.username = email
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid Email or Password!")

# ================================
# Logout Button
# ================================
def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("👋 Logged out successfully!")

# ================================
# Main App
# ================================
st.sidebar.title("Navigation")

if not st.session_state.logged_in:
    page = st.sidebar.radio("Go to", ["Login", "Sign Up"])
    if page == "Login":
        login_page()
    else:
        signup_page()

else:
    st.title("🩺 Chronic Kidney Disease Prediction System")
    st.subheader(f"Welcome, {st.session_state.username} 🎉")
    st.write("Enter patient details carefully to predict CKD condition.")

    st.markdown("### 🧍 Demographic Details")
    age = st.number_input("Age (years)", 1, 100, help="Enter patient's age (1–100 years)")

    st.markdown("### 💓 Vital Signs")
    bp = st.number_input("Blood Pressure (mmHg)", 50, 200, help="Normal: 90–120 (Systolic) / 60–80 (Diastolic)")

    st.markdown("### 🧫 Urine Test Parameters")
    sg = st.number_input("Specific Gravity", 1.000, 1.040, step=0.001,
                         help="Normal: 1.005–1.030 (indicates urine concentration)")
    al = st.number_input("Albumin (g/dL)", 0, 5,
                         help="Normal: 0–1; High (>2) indicates kidney damage")
    su = st.number_input("Sugar (Urine Glucose Level)", 0, 5,
                         help="Urine sugar levels: 0=None (Normal), 1=Trace, 2=+, 3=++, 4=+++, 5=++++ (High levels indicate diabetes risk)")

    st.markdown("### 💉 Blood Test Parameters")
    bgr = st.number_input("Blood Glucose Random (mg/dL)", 50, 500,
                          help="Normal: 70–140 mg/dL; >200 mg/dL suggests diabetes")
    sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 15.0,
                         help="Normal: 0.6–1.2 mg/dL; High indicates kidney dysfunction")
    sod = st.number_input("Sodium (mEq/L)", 100, 200,
                          help="Normal: 135–145 mEq/L; imbalance affects kidney function")
    pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0,
                          help="Normal: 3.5–5.0 mEq/L; abnormal in CKD")
    hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 17.0,
                           help="Normal: 12–16 g/dL; low indicates anemia (common in CKD)")
    pcv = st.number_input("Packed Cell Volume (%)", 20, 60,
                          help="Normal: 36–50%; indicates red blood cell percentage")
    wc = st.number_input("White Blood Cell Count (cells/µL)", 2000, 30000,
                         help="Normal: 4000–11000; high indicates infection or inflammation")

    # Prepare features
    features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
    features = scaler.transform(features)

    st.markdown("### 🤖 Choose Prediction Model")
    model_choice = st.selectbox("Select a Model", list(models.keys()))

    if st.button("Predict CKD"):
        model = models[model_choice]
        prediction = model.predict(features)[0]
        result = "🩸 CKD Detected" if prediction == 1 else "✅ No CKD Detected"
        st.success(f"Prediction Result: {result}")

    st.write("---")
    logout_button()



