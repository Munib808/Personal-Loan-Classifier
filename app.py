import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
import json


# Load model and scaler
with open("voting_model_personal_loan.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler_voting_model_personal_loan.pkl", "rb") as file:
    scaler = pickle.load(file)

# Columns in same order as training
columns = ['age', 'balance', 'campaign', 'pdays', 'previous', 'job_admin.',
           'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
           'job_management', 'job_retired', 'job_self-employed', 'job_services',
           'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
           'marital_divorced', 'marital_married', 'marital_single',
           'education_primary', 'education_secondary', 'education_tertiary',
           'education_unknown', 'default_no', 'default_yes', 'housing_no',
           'housing_yes', 'loan_no', 'loan_yes', 'contact_cellular',
           'contact_telephone', 'contact_unknown', 'poutcome_failure',
           'poutcome_other', 'poutcome_success', 'poutcome_unknown']

# Page Setup
st.set_page_config(page_title="Loan Approval Predictor", layout="centered", page_icon="🏦")
st.title("🏦 Bank Personal Loan Prediction App")
st.subheader("Will the client subscribe to a term deposit?")

st.markdown("### 📝 Enter Client Information:")

age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance (€)", -5000, 100000, 1000)
campaign = st.number_input("Number of contacts performed during this campaign and for this client (numeric, includes last contact)", 1, 50, 1)
pdays = st.number_input("Number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)", -1, 999, -1)
previous = st.number_input("Number of contacts performed before this campaign and for this client (numeric)", 0, 50, 0)

job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician',
                           'unemployed', 'unknown'])

marital = st.selectbox("Marital Status note: divorced means divorced or widowed", ['divorced', 'married', 'single'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Has credit in default?", ['no', 'yes'])
housing = st.selectbox("Has housing loan?", ['no', 'yes'])
loan = st.selectbox("Has personal loan?", ['no', 'yes'])
contact = st.selectbox("Contact communication type", ['cellular', 'telephone', 'unknown'])
poutcome = st.selectbox("Previous campaign outcome", ['failure', 'other', 'success', 'unknown'])

# Construct input row
input_data = {col: 0 for col in columns}
input_data["age"] = age
input_data["balance"] = balance
input_data["campaign"] = campaign
input_data["pdays"] = pdays
input_data["previous"] = previous

input_data[f"job_{job}"] = 1
input_data[f"marital_{marital}"] = 1
input_data[f"education_{education}"] = 1
input_data[f"default_{default}"] = 1
input_data[f"housing_{housing}"] = 1
input_data[f"loan_{loan}"] = 1
input_data[f"contact_{contact}"] = 1
input_data[f"poutcome_{poutcome}"] = 1

# Convert to DataFrame
X_input = pd.DataFrame([input_data])

# Scale input
X_scaled = scaler.transform(X_input)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(X_scaled)[0]

    if prediction == 1:
        st.success("✅ Client is likely to subscribe to the term deposit.")
    else:
        st.error("❌ Client is not likely to subscribe to the term deposit.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ by Munib | Model: Voting Classifier | Deployment: Streamlit")
