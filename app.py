import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the Random Forest model
with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Insurance Cost Prediction App")
st.write("This app predicts the insurance cost based on input details using a Random Forest Regressor.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Encode categorical values
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0

# One-hot encode region
region_vals = {
    "southeast": [1, 0, 0, 0],
    "southwest": [0, 1, 0, 0],
    "northeast": [0, 0, 1, 0],
    "northwest": [0, 0, 0, 1]
}
region_encoded = region_vals[region]

# Combine all features
input_data = [age, sex_val, bmi, children, smoker_val] + region_encoded
input_df = pd.DataFrame([input_data], columns=[
    "age", "sex", "bmi", "children", "smoker",
    "region_southeast", "region_southwest", "region_northeast", "region_northwest"
])

# Predict
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
