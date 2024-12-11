# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:41:49 2024

@author: whynew.in
"""

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

# Load necessary data
soil_data_df = pd.read_excel('05-Macro and Micro_soil_nutrient_MAH6july24.xlsx')
binary_encoded_variety_df = pd.read_csv('unique_varieties_binary_encoded.csv')
source_class_df = pd.read_csv('Source Class_encoded_values.csv')
dest_class_df = pd.read_csv('Destination Class_encoded_values.csv')
sowing_week_df = pd.read_csv('Sowing Week_encoded_values.csv')
sowing_month_df = pd.read_csv('Sowing Month_encoded_values.csv')
harvest_week_df = pd.read_csv('Harvesting Week_encoded_values.csv')
harvest_month_df = pd.read_csv('Harvesting Month_encoded_values.csv')
crop_pattern_df = pd.read_csv('cropPattern_encoded_values.csv')
typeofsowing_df = pd.read_csv('typeOfSowing_encoded_values.csv')
crop_variety_df = pd.read_csv('MH_crops_and_varieties.csv')

# Load trained model
model = load_model('ALL_Fea_model_27Jun_f84_512_150_3.h5')

# Function to get user input
def get_user_input():
    user_input = {
        'Certified Area': st.number_input("Enter Certified Area (ha)", value=0.0, min_value=0.0),
        'sowedQuantity': st.number_input("Enter sowedQuantity (q)", value=0.0, min_value=0.0),
        'Source Class': st.selectbox("Enter Source Class", source_class_df['Source Class'].unique()),
        'Destination Class': st.selectbox("Enter Destination Class", dest_class_df['Destination Class'].unique()),
        'Sowing Week': st.selectbox("Enter Sowing Week", sowing_week_df['Sowing Week'].unique()),
        'Sowing Month': st.selectbox("Enter Sowing Month", sowing_month_df['Sowing Month'].unique()),
        'Harvesting Week': st.selectbox("Enter Harvesting Week", harvest_week_df['Harvesting Week'].unique()),
        'Harvesting Month': st.selectbox("Enter Harvesting Month", harvest_month_df['Harvesting Month'].unique()),
        'cropPattern': st.selectbox("Enter cropPattern", crop_pattern_df['cropPattern'].unique()),
        'typeOfSowing': st.selectbox("Enter typeOfSowing", typeofsowing_df['typeOfSowing'].unique())
    }

    selected_crop = st.selectbox("Enter Crop", crop_variety_df['Crop'].unique())
    varieties = crop_variety_df[crop_variety_df['Crop'] == selected_crop]['Variety'].unique()
    user_input['Variety'] = st.selectbox("Enter Variety", varieties)
    user_input['District'] = st.selectbox("Enter District", soil_data_df['District'].unique())

    return user_input

# Function to fetch soil parameters for the district
def get_soil_parameters(district):
    soil_params = soil_data_df[soil_data_df['District'] == district].iloc[0]  # Assuming district is unique
    return soil_params.values[1:]  # Exclude District column

# Function to get binary encoded variety
def get_binary_encoded_variety(variety_name):
    row = binary_encoded_variety_df[binary_encoded_variety_df['Variety'] == variety_name]
    if not row.empty:
        binary_encoded_values = row.select_dtypes(include=[np.number]).values.astype(float).flatten()
    else:
        binary_encoded_values = np.zeros(len(binary_encoded_variety_df.columns) - 1)
    return binary_encoded_values

# Function to get one-hot encoded value for a column
def get_one_hot_encoded_value(column_value, encoding_df):
    row = encoding_df[encoding_df.iloc[:, 0] == column_value]
    if not row.empty:
        encoded_values = row.select_dtypes(include=[np.number]).values.astype(float).flatten()
    else:
        encoded_values = np.zeros(len(encoding_df.columns) - 1)
    return encoded_values

# Main function to make predictions
def predict_yield(user_input):
    soil_params = get_soil_parameters(user_input['District'])
    binary_encoded_values = get_binary_encoded_variety(user_input['Variety'])
    
    source_class_encoded = get_one_hot_encoded_value(user_input['Source Class'], source_class_df)
    destination_class_encoded = get_one_hot_encoded_value(user_input['Destination Class'], dest_class_df)
    sowing_week_encoded = get_one_hot_encoded_value(user_input['Sowing Week'], sowing_week_df)
    sowing_month_encoded = get_one_hot_encoded_value(user_input['Sowing Month'], sowing_month_df)
    harvest_week_encoded = get_one_hot_encoded_value(user_input['Harvesting Week'], harvest_week_df)
    harvest_month_encoded = get_one_hot_encoded_value(user_input['Harvesting Month'], harvest_month_df)
    crop_pattern_encoded = get_one_hot_encoded_value(user_input['cropPattern'], crop_pattern_df)
    type_of_sowing_encoded = get_one_hot_encoded_value(user_input['typeOfSowing'], typeofsowing_df)
    
    source_class_encoded = np.delete(source_class_encoded, 0)
    destination_class_encoded = np.delete(destination_class_encoded, 0)
    sowing_week_encoded = np.delete(sowing_week_encoded, 0)
    sowing_month_encoded = np.delete(sowing_month_encoded, 0)
    harvest_week_encoded = np.delete(harvest_week_encoded, 0)
    harvest_month_encoded = np.delete(harvest_month_encoded, 0)
    crop_pattern_encoded = np.delete(crop_pattern_encoded, 0)
    type_of_sowing_encoded = np.delete(type_of_sowing_encoded, 0)
    
    input_features = np.concatenate([
        [user_input['Certified Area']],
        [user_input['sowedQuantity']],
        soil_params,
        source_class_encoded,
        destination_class_encoded,
        sowing_week_encoded,
        sowing_month_encoded,
        harvest_week_encoded,
        harvest_month_encoded,
        crop_pattern_encoded,
        type_of_sowing_encoded,
        binary_encoded_values
    ])
    
    input_features = input_features.astype(np.float32).reshape(1, -1)
    predicted_yield = model.predict(input_features)
    
    return predicted_yield

# Streamlit UI
st.title("SATHI - Crop Estimated Yield Prediction")
st.image("SATHI_Logo.png", width=150)

user_input = get_user_input()

if st.button("Predict Yield"):
    if user_input['Certified Area'] <= 0 or user_input['sowedQuantity'] <= 0:
        st.warning("Certified Area and sowedQuantity must be greater than zero.")
    else:
        predicted_yield = predict_yield(user_input)
        st.subheader("Prediction")
        st.success(f"Predicted Yield: {predicted_yield[0][0]:.2f} units")
