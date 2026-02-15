import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

Page setup

st.set_page_config(page_title="AI Disease Prediction", page_icon="🏥")

st.title("🏥 AI-Based Disease Prediction System")
st.write("Enter patient health details to predict diabetes risk")

Get current directory safely (works on Streamlit Cloud)

BASE_DIR = os.path.dirname(os.path.abspath(file))

File paths

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
accuracy_path = os.path.join(BASE_DIR, "accuracy.pkl")

Load files safely

try:
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
accuracy = joblib.load(accuracy_path)
except Exception as e:
st.error("❌ Model files not found or incompatible.")
st.error("Make sure model.pkl, scaler.pkl, accuracy.pkl are in the same folder as app.py")
st.stop()

Show model accuracy

st.info(f"Model Accuracy: {accuracy*100:.2f}%")

User inputs

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

Prediction

if st.button("Predict Disease Risk"):

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")

if probability > 0.7:
    st.error(f"🔴 High Risk of Diabetes ({probability*100:.2f}%)")
    st.write("⚠️ Consult doctor, maintain diet, exercise regularly.")
elif probability > 0.4:
    st.warning(f"🟠 Moderate Risk ({probability*100:.2f}%)")
    st.write("⚠️ Monitor health and schedule checkups.")
else:
    st.success(f"🟢 Low Risk ({(1-probability)*100:.2f}%)")
    st.write("✅ Maintain healthy lifestyle.")

# Chart
fig, ax = plt.subplots()
ax.bar(["Risk %"], [probability*100])
ax.set_ylabel("Percentage")
ax.set_title("Diabetes Risk Level")
st.pyplot(fig)

st.write("---")
st.caption("This project is for educational purposes only and not a medical diagnosis tool.")
