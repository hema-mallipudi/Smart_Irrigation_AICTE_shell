import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Smart Sprinkler System", page_icon="💧", layout="centered")

# Load model
@st.cache_resource(show_spinner=True)
def load_model():
    return joblib.load("Farm_Irrigation_System.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'Farm_Irrigation_System.pkl' is present.")
    st.stop()

# Title & Instructions
st.title("💧 Smart Sprinkler System")
st.markdown("#### Enter scaled sensor values (0 to 1) to predict sprinkler status")
st.divider()

# Sensor Inputs (20 sliders in 10 rows)
sensor_values = []
for i in range(0, 20, 2):
    col1, col2 = st.columns(2)
    with col1:
        val1 = st.slider(f"Sensor {i}", 0.0, 1.0, 0.5, 0.01)
        sensor_values.append(val1)
    with col2:
        val2 = st.slider(f"Sensor {i+1}", 0.0, 1.0, 0.5, 0.01)
        sensor_values.append(val2)

# Predict button
if st.button("🔍 Predict Sprinklers"):
    input_array = np.array(sensor_values).reshape(1, -1)

    try:
        prediction = model.predict(input_array)[0]
        st.success("✅ Prediction Complete!")

        st.markdown("### 🌿 Sprinkler Status:")
        for i, status in enumerate(prediction):
            emoji = "🟢 ON" if status == 1 else "🔴 OFF"
            st.write(f"**Sprinkler {i}**: {emoji}")

        st.info(f"ℹ️ Note: This model predicts for only {len(prediction)} sprinklers.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
