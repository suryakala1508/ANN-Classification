import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Load model, encoders, scaler
# ----------------------------
model = tf.keras.models.load_model('/home/rgukt/Desktop/ANN classification/model.h5')

with open('/home/rgukt/Desktop/ANN classification/label_encoder_gender', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('/home/rgukt/Desktop/ANN classification/onehot_encoder_geo', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('/home/rgukt/Desktop/ANN classification/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction Dashboard")

# Sidebar Inputs
st.sidebar.header("Customer Input Data")
def user_input_features():
    geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
    age = st.sidebar.slider('Age', 18, 92, 30)
    balance = st.sidebar.number_input('Balance', 0, 250000, 50000)
    credit_score = st.sidebar.number_input('Credit Score', 300, 850, 600)
    estimated_salary = st.sidebar.number_input('Estimated Salary', 0, 200000, 50000)
    tenure = st.sidebar.slider('Tenure', 0, 10, 3)
    num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])
    
    data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # ----------------------------
    # One-hot encode Geography
    # ----------------------------
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    data = pd.concat([data.reset_index(drop=True), geo_encoded_df], axis=1)

    return data

input_data = user_input_features()

# ----------------------------
# Scale the input
# ----------------------------
input_data_scaled = scaler.transform(input_data)

# ----------------------------
# Individual Prediction
# ----------------------------
st.subheader("Individual Customer Prediction")
prediction = model.predict(input_data_scaled)
churn_proba = prediction[0][0]

color = 'green' if churn_proba < 0.4 else 'yellow' if churn_proba < 0.7 else 'red'
st.markdown(f"**Churn Probability:** <span style='color:{color}; font-size:24px'>{churn_proba:.2f}</span>", unsafe_allow_html=True)

if churn_proba > 0.5:
    st.warning("The customer is likely to churn! Consider retention strategies.")
else:
    st.success("The customer is not likely to churn.")

# ----------------------------
# SHAP Explainability
# ----------------------------
st.subheader("Feature Contribution (SHAP)")
explainer = shap.KernelExplainer(model.predict, input_data_scaled)
shap_values = explainer.shap_values(input_data_scaled, nsamples=100)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, input_data, show=False)
st.pyplot(fig)

# ----------------------------
# Batch Prediction
# ----------------------------
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    
    # Preprocessing batch data
    batch_data['Gender'] = label_encoder_gender.transform(batch_data['Gender'])
    
    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(batch_data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    batch_data = pd.concat([batch_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Scale batch data
    batch_data_scaled = scaler.transform(batch_data)
    
    # Predict
    batch_pred = model.predict(batch_data_scaled)
    batch_data['Churn_Probability'] = batch_pred
    st.dataframe(batch_data.sort_values(by='Churn_Probability', ascending=False))
    
    # Download batch results
    csv = batch_data.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")
'
# ----------------------------
# Dashboard Insights
# ----------------------------
st.subheader("Dashboard Insights")
st.write("Visualizing predicted churn probabilities.")

# Histogram for single prediction (optional)
fig2, ax2 = plt.subplots()
ax2.hist([churn_proba], bins=10, color='skyblue', edgecolor='black')
ax2.set_xlabel("Churn Probability")
ax2.set_ylabel("Number of Customers")
ax2.set_title("Churn Probability Distribution")
st.pyplot(fig2)

st.write("✅ This enhanced dashboard shows individual & batch predictions, feature contributions (SHAP), and visual insights for recruiters or business stakeholders.")
