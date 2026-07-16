import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Load Model and Label Encoder
# ----------------------------
with open("crop_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Crop Recommendation Agent",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Recommendation AI Agent")
st.write("Enter the soil and environmental conditions to get the best crop recommendation.")

st.markdown("---")

# ----------------------------
# User Inputs
# ----------------------------

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)

P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)

K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=25.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

ph = st.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=200.0
)

st.markdown("---")

# ----------------------------
# Prediction
# ----------------------------

if st.button("Recommend Crop"):

    sample = np.array([[N, P, K,
                        temperature,
                        humidity,
                        ph,
                        rainfall]])

    prediction = model.predict(sample)

    crop = label_encoder.inverse_transform(prediction)

    st.success(f"✅ Recommended Crop: **{crop[0]}**")

    st.balloons()

    st.markdown("### 🌱 Recommendation Summary")

    st.write(f"""
    - Nitrogen : **{N}**
    - Phosphorus : **{P}**
    - Potassium : **{K}**
    - Temperature : **{temperature} °C**
    - Humidity : **{humidity}%**
    - Soil pH : **{ph}**
    - Rainfall : **{rainfall} mm**
    """)

    st.info(f"The AI model recommends **{crop[0]}** as the most suitable crop for the given soil and weather conditions.")