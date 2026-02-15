import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
accuracy = joblib.load("accuracy.pkl")

st.set_page_config(page_title="AI Disease Prediction", page_icon="🏥")


st.title("🏥 AI-Based Disease Prediction System")
st.write("Enter patient details to predict diabetes risk")


st.info(f"Model Accuracy: {accuracy*100:.2f}%")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 100)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

if st.button("Predict Disease Risk"):
input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")

if probability > 0.7:
    st.error(f"🔴 High Risk of Diabetes ({probability*100:.2f}%)")
    st.write("⚠️ Recommended: Consult doctor, improve diet, increase physical activity.")
elif probability > 0.4:
    st.warning(f"🟠 Moderate Risk ({probability*100:.2f}%)")
    st.write("⚠️ Recommended: Regular checkups, lifestyle monitoring.")
else:
    st.success(f"🟢 Low Risk ({(1-probability)*100:.2f}%)")
    st.wr

