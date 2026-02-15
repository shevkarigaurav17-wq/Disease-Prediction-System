import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Disease Prediction", page_icon="🏥")

st.title("🏥 AI-Based Disease Prediction System")
st.write("Enter patient health details to predict diabetes risk")

# Load model files (must be in same folder as app.py)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
accuracy = joblib.load("accuracy.pkl")

st.info(f"Model Accuracy: {accuracy*100:.2f}%")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

if st.button("Predict Disease Risk"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data = scaler.transform(data)
    prob = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prob > 0.7:
        st.error(f"High Risk of Diabetes ({prob*100:.2f}%)")
    elif prob > 0.4:
        st.warning(f"Moderate Risk ({prob*100:.2f}%)")
    else:
        st.success(f"Low Risk ({(1-prob)*100:.2f}%)")

    fig, ax = plt.subplots()
    ax.bar(["Risk %"], [prob*100])
    st.pyplot(fig)

st.caption("For educational purposes only.")
