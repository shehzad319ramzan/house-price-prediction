import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------
# PAGE CONFIG (UI SETUP)
# -------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------------
# LOAD MODEL + SCALER
# -------------------------
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# HEADER SECTION
# -------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")
st.markdown("---")

# -------------------------
# SIDEBAR (INFO PANEL)
# -------------------------
st.sidebar.title("ℹ️ Project Info")
st.sidebar.info("This app uses Ridge Regression to predict house prices based on input features.")


# -------------------------
# INPUT SECTION (COLUMNS UI)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)

with col2:
    stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

st.markdown("---")

# -------------------------
# PREDICTION BUTTON
# -------------------------
predict_btn = st.button("🔮 Predict House Price")

# -------------------------
# PREDICTION LOGIC
# -------------------------
if predict_btn:

    # Create DataFrame (IMPORTANT for scaler compatibility)
    input_data = pd.DataFrame([[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking
    ]], columns=['area','bedrooms','bathrooms','stories','parking'])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # -------------------------
    # RESULT DISPLAY (STYLISH BOX)
    # -------------------------
    st.markdown("### 🎯 Prediction Result")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #1f77b4, #4a90e2);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 26px;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        ">
        💰 Estimated Price: {prediction[0]:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("💡 Built with Streamlit | ML Project for Learning")