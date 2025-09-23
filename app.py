import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🩺 Chronic Kidney Disease Prediction")

# Load models
scaler = joblib.load("models/scaler.pkl")
models = {
    "Logistic Regression": joblib.load("models/logistic.pkl"),
    "Random Forest": joblib.load("models/randomforest.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naivebayes.pkl")
}

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
# rc = st.number_input("Red Blood Cell Count", 2.0, 8.0) # Removed this line


# features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc, rc]]) # Modified this line
features = np.array([[age, bp, sg, al, su, bgr, sc, sod, pot, hemo, pcv, wc]])
features = scaler.transform(features)

model_choice = st.selectbox("Choose Model", list(models.keys()))

if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(features)[0]
    result = "CKD Detected" if prediction == 1 else "No CKD"
    st.success(f"Prediction: {result}")
