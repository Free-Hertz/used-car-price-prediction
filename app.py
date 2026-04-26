import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load data + model
df = pd.read_csv("data/processed/vehicles_feature_audited.csv")
model = joblib.load("models/final_lightgbm_pipeline.pkl")

# Title
st.markdown("<h1 style='text-align: center;'>🚗 Used Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter car details to estimate price</p>", unsafe_allow_html=True)
st.markdown("---")

# Dropdown values
manufacturers = sorted(df["manufacturer"].dropna().unique())
conditions = sorted(df["condition"].dropna().unique())
cylinders_list = sorted(df["cylinders"].dropna().unique())
fuel_types = sorted(df["fuel"].dropna().unique())
title_statuses = sorted(df["title_status"].dropna().unique())
transmissions = sorted(df["transmission"].dropna().unique())
drives = sorted(df["drive"].dropna().unique())
sizes = sorted(df["size"].dropna().unique())
types = sorted(df["type"].dropna().unique())
colors = sorted(df["paint_color"].dropna().unique())

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=2015
    )
    
    manufacturer = st.selectbox("Manufacturer", manufacturers)
    condition = st.selectbox("Condition", conditions)
    fuel = st.selectbox("Fuel", fuel_types)
    transmission = st.selectbox("Transmission", transmissions)
    size = st.selectbox("Size", sizes)

with col2:
    odometer = st.number_input(
        "Odometer",
        min_value=0,
        max_value=int(df["odometer"].max()),
        value=80000
    )
    
    model_name = st.text_input("Model", "camry")
    cylinders = st.selectbox("Cylinders", cylinders_list)
    drive = st.selectbox("Drive", drives)
    type_ = st.selectbox("Type", types)
    paint_color = st.selectbox("Paint Color", colors)

# Remaining fields
st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    title_status = st.selectbox("Title Status", title_statuses)

with col4:
    region = st.text_input("Region", "los angeles")

# Centered button
st.markdown(" ")
_, center_col, _ = st.columns([1,2,1])

with center_col:
    predict_btn = st.button("Predict Price")

# Prediction
if predict_btn:
    input_df = pd.DataFrame([{
        'year': year,
        'odometer': odometer,
        'manufacturer': manufacturer,
        'model': model_name,
        'condition': condition,
        'cylinders': cylinders,
        'fuel': fuel,
        'title_status': title_status,
        'transmission': transmission,
        'drive': drive,
        'size': size,
        'type': type_,
        'paint_color': paint_color,
        'region': region
    }])

    pred_log = model.predict(input_df)
    pred_price = np.exp(pred_log)[0]

    st.markdown(f"""
    <div style='text-align:center; padding:20px; background-color:#1e3a2f; border-radius:10px;'>
        <h3 style='color:#4ade80;'>Estimated Price: ${pred_price:,.2f}</h3>
    </div>
    """, unsafe_allow_html=True)