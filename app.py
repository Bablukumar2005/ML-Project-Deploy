import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model with error handling
try:
    with open("rf_model.pkl", "rb") as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Error: The model file 'rf_model.pkl' was not found. Please ensure it is in the correct directory.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Error: Missing required module: {str(e)}. Ensure 'scikit-learn' is installed with the correct version.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

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

# Encode region numerically (matching training preprocessing)
region_map = {
    "southeast": 0,
    "southwest": 1,
    "northeast": 2,
    "northwest": 3
}
region_val = region_map[region]

# Combine all features
input_data = [age, sex_val, bmi, children, smoker_val, region_val]
input_df = pd.DataFrame([input_data], columns=[
    "age", "sex", "bmi", "children", "smoker", "region"
])

# Predict
if st.button("Predict Insurance Cost"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")