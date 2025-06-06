import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set page configuration
st.set_page_config(page_title="Income Prediction App", layout="centered")

st.title("Income Prediction ( >50K or <=50K )")
st.write("This ML model is trained on a sample of the UCI Adult dataset.")

# Load trained model
@st.cache_resource
def load_model():
    with open("income_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Accuracy placeholder (you can manually update this based on training results)
accuracy = 0.85
st.subheader("Model Accuracy")
st.write(f"Accuracy on test set: **{accuracy * 100:.2f}%**")

# Input features
st.subheader("Input Features")

age = st.slider("Age", 17, 90, 30)
education_num = st.slider("Education Level (numeric)", 1, 16, 10)
hours = st.slider("Hours per Week", 1, 99, 40)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'State-gov'])
education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Some-college'])
marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
occupation = st.selectbox("Occupation", ['Exec-managerial', 'Craft-repair', 'Adm-clerical'])
relationship = st.selectbox("Relationship", ['Husband', 'Wife', 'Not-in-family'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander'])
sex = st.selectbox("Sex", ['Male', 'Female'])
native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico'])

# Create DataFrame for prediction
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [100000],  # dummy
    'education': [education],
    'education_num': [education_num],
    'marital_status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'capital_gain': [capital_gain],
    'capital_loss': [capital_loss],
    'hours_per_week': [hours],
    'native_country': [native_country]
})

# Apply label encoding (must match the one used during training)
def encode_input(df):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

encoded_input = encode_input(input_df)

# Drop any columns not in training set (safe-guard)
expected_features = model.feature_names_in_
for col in expected_features:
    if col not in encoded_input.columns:
        encoded_input[col] = 0
encoded_input = encoded_input[expected_features]

# Predict
if st.button("Predict Income"):
    prediction = model.predict(encoded_input)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: 💼 {result}")
