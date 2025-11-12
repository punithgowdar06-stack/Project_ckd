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
# User Data Management
# ================================
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# Load users persistently
if "users" not in st.session_state:
    st.session_state.users = load_users()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ================================
# Pages: Sign Up / Login / Logout
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
            save_users(st.session_state.users)
            st.success("✅ Account created successfully! You can now log in.")

def login_page():
    st.title("🔐 Login Page")
    email = st.text_input("Email ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = st.session_state.users
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
# Health Abnormality Checker
# ================================
def check_abnormalities(values):
    age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc = values
    messages = []

    if bp > 140:
        messages.append("⚠️ High Blood Pressure (>140 mmHg) — can strain kidneys.")
    elif bp < 80:
        messages.append("⚠️ Low Blood Pressure (<80 mmHg) — can reduce kidney perfusion.")

    if sg < 1.005:
        messages.append("⚠️ Low Specific Gravity — possible kidney filtration issue.")

    if al > 1:
        messages.append("⚠️ High Albumin — protein in urine (proteinuria), early CKD indicator.")

    if su > 150:
        messages.append("⚠️ High Urine Sugar (>150 mg/dL) — possible diabetes risk.")

    if bgr > 200:
        messages.append("⚠️ High Blood Glucose (>200 mg/dL) — diabetes may affect kidneys.")
    elif bgr < 70:
        messages.append("⚠️ Low Blood Glucose (<70 mg/dL).")

    if sc > 1.5:
        messages.append("⚠️ High Serum Creatinine — reduced kidney function.")
    elif sc < 0.6:
        messages.append("⚠️ Low Serum Creatinine — possible muscle loss.")

    if sod < 130:
        messages.append("⚠️ Low Sodium — possible electrolyte imbalance.")
    elif sod > 145:
        messages.append("⚠️ High Sodium — dehydration risk.")

    if pot > 6:
        messages.append("⚠️ High Potassium — heart risk, common in CKD.")
    elif pot < 3.5:
        messages.append("⚠️ Low Potassium — muscle weakness.")

    if hemo < 11:
        messages.append("⚠️ Low Hemoglobin — anemia (common in CKD).")

    if pcv < 35:
        messages.append("⚠️ Low PCV — anemia or CKD.")
    elif pcv > 50:
        messages.append("⚠️ High PCV — dehydration or heart condition.")

    if wc > 11000:
        messages.append("⚠️ High WBC Count — infection/inflammation.")
    elif wc < 4000:
        messages.append("⚠️ Low WBC Count — immune issue.")

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

    st.header("📋 Demographic Details")
    age = st.number_input("🧍 Age (years)", min_value=1, max_value=100, value=30)
    st.caption("✅ Normal: Any age (CKD risk rises after 40 years)")

    st.header("❤️ Vital Signs")
    bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=180, value=120)
    st.caption("✅ Normal: 90–120 | ⚠️ High: >140")

    st.header("🧪 Urine Test Parameters")
    sg = st.number_input("Specific Gravity", min_value=1.00, max_value=1.03, value=1.02, step=0.001)
    st.caption("✅ Normal: 1.005–1.025")
    al = st.number_input("Albumin (g/dL)", min_value=0, max_value=5, value=1)
    st.caption("✅ Normal: 0 | ⚠️ High: >1 indicates protein in urine")
    su = st.number_input("Urine Sugar Level (mg/dL)", min_value=0, max_value=400, value=80)
    st.caption("✅ Normal:70-100 | ⚠️ High: >126 may indicate diabetes")

    st.header("🩸 Blood Test Parameters")
    bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=70, max_value=490, value=120)
    st.caption("✅ Normal: 70–140 | ⚠️ High: >200 may indicate diabetes")
    sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.4, max_value=15.0, value=1.2)
    st.caption("✅ Normal: 0.6–1.2 | ⚠️ High: >1.5 may indicate kidney damage")
    sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=150, value=135)
    st.caption("✅ Normal: 135–145")
    pot = st.number_input("Potassium (mEq/L)", min_value=2.5, max_value=7.0, value=4.5)
    st.caption("✅ Normal: 3.5–5.5")
    hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=17.0, value=13.5)
    st.caption("✅ Normal: 12–17")
    pcv = st.number_input("Packed Cell Volume (%)", min_value=20, max_value=60, value=40)
    st.caption("✅ Normal: 35–50")
    wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=2000, max_value=30000, value=8000)
    st.caption("✅ Normal: 4000–11000")
    # ----------------
    # Prediction Section
    # ----------------
    features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
    features_scaled = scaler.transform(features)
    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("🔍 Predict CKD"):
        model = models[model_choice]
        prediction = model.predict(features_scaled)[0]
        reasons = check_abnormalities([age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc])

        if prediction == 1:
            st.error("🚨 **Prediction: CKD Detected!**")
            if reasons:
                st.subheader("🔎 Possible Causes Leading to CKD:")
                for msg in reasons:
                    st.write(msg)
                st.warning("💊 These conditions are enough to cause CKD. Please consult a nephrologist immediately and start treatment early.")
            else:
                st.info("⚠️ CKD detected, but no specific abnormal cause identified. Consult a doctor for detailed tests.")
        else:
            if len(reasons) > 0:
                st.warning("⚠️ **No CKD detected, but some parameters are abnormal!**")
                for msg in reasons:
                    st.write(msg)
                st.info("🩺 These could be early signs of kidney or metabolic issues. Please consult a doctor for further evaluation.")
            else:
                st.success("✅ **No CKD Detected! All test parameters are normal. Stay healthy!**")

    st.write("---")
    logout_button()
