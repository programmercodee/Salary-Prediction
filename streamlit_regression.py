import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder

#load the ann train model
model = tf.keras.models.load_model('regression_model.h5')

## load the scaler pickle , onehot
with open('salary_label_encoder_gender.pkl' ,'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('salary_onehot_encoder_geo.pkl' ,'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('salary_scaler.pkl' ,'rb') as file:
    scaler = pickle.load(file)
    
st.title("Salary Prediction")  

geography = st.selectbox("Select Geography", label_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Select Age", 18 , 99)
balance = st.number_input("Enter Balance", min_value=0.0, step=100.0)
credit_score = st.number_input("Enter Credit Score", min_value=0, step=100, max_value=850)
tenure = st.slider("Select Tenure", 0, 10)
exited = st.selectbox("Exited", [0, 1])
num_of_products = st.slider("Select Number of Products", 0, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'Tenure': [tenure],
    'Exited': [exited],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = input_data[scaler.feature_names_in_]
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
predicted_proba = prediction[0][0]

# st.write("Predicted Salary: ${predicted_proba:.2f}")
st.write(f"Predicted Salary: Rs{predicted_proba:.2f}")
st.write("Input Data after reordering:", input_data)