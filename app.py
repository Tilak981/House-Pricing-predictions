import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("linear_regression_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction")
st.markdown("Enter the details of the house below:")

# -----------------------
# Numeric features in 2 columns
# -----------------------
col1, col2 = st.columns(2)
with col1:
    area = st.slider("Area (sq ft)", 100, 10000, 1000)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
with col2:
    stories = st.slider("Stories", 1, 5, 1)

# -----------------------
# Binary features in 2 columns
# -----------------------
def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0

col3, col4 = st.columns(2)
with col3:
    mainroad = yes_no_to_binary(st.selectbox("Main Road", ["No", "Yes"]))
    guestroom = yes_no_to_binary(st.selectbox("Guest Room", ["No", "Yes"]))
    basement = yes_no_to_binary(st.selectbox("Basement", ["No", "Yes"]))
with col4:
    hotwaterheating = yes_no_to_binary(st.selectbox("Hot Water Heating", ["No", "Yes"]))
    airconditioning = yes_no_to_binary(st.selectbox("Air Conditioning", ["No", "Yes"]))
    prefarea = yes_no_to_binary(st.selectbox("Preferred Area", ["No", "Yes"]))

# -----------------------
# Furnishing Status
# -----------------------
furnishingstatus = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
furnishingstatus_semi_furnished = 1 if furnishingstatus == "Semi-Furnished" else 0
furnishingstatus_furnished = 1 if furnishingstatus == "Furnished" else 0

# Build input DataFrame

input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "prefarea": prefarea,
    "furnishingstatus_semi-furnished": furnishingstatus_semi_furnished,
    "furnishingstatus_furnished": furnishingstatus_furnished
}

input_df = pd.DataFrame([input_dict])
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]


# USD → INR conversion

usd_to_inr = 83


# Predict Button

if st.button("💰 Predict House Price", use_container_width=True):
    prediction = model.predict(input_df)
    prediction_inr = prediction[0] * usd_to_inr
    st.success(f"Predicted House Price: ₹{prediction_inr:,.2f}")


# Credit

st.markdown("---")
st.markdown("**Model created by Tanishq**")

