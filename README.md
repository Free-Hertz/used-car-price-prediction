# Used Car Price Prediction

This project builds an end-to-end machine learning pipeline to estimate fair prices for used cars using real-world listing data.

The goal was not just to train a model, but to go through the full ML workflow — from cleaning messy data to deploying a working app.

---

## Live Demo

https://used-car-price-prediction-b9q1.onrender.com/

---

## Problem

Used car listings vary a lot in price, even for similar vehicles. Sellers often:

- Overprice → cars don’t sell  
- Underprice → lose money  
- Misprice → reduce buyer trust  

This project aims to predict a **reasonable market price** based on vehicle details.

---

## Dataset

- Source: Craigslist Cars & Trucks Dataset (Kaggle)  
- Original size: ~426k rows  
- Final dataset: ~390k rows after cleaning  

Original dataset: [Craigslist Cars and Trucks Data - Austin Rees (Kaggle) : https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data]

Features include:
- Year, Odometer  
- Manufacturer, Model  
- Fuel, Transmission, Condition  
- Region and more  

---

## Data Processing

Some key steps:

- Removed invalid listings (e.g. price = 0)  
- Trimmed extreme outliers  
- Dropped leakage columns (IDs, URLs, etc.)  
- Applied **log transformation on price** to handle skew  

---

## Feature Engineering

- **Numerical:** year, odometer  
- **High-cardinality categorical (model, region):**
  - handled using **Target Encoding**
- **Other categorical features:**
  - encoded using One-Hot Encoding  

All preprocessing was built using a **scikit-learn pipeline**.

---

## Model

Models tried:
- Linear Regression  
- Random Forest  
- LightGBM  

Final model: **LightGBM Regressor**

Tree-based models performed much better, capturing non-linear relationships in pricing.

---

## Results

- Final Log RMSE: **~0.71**  
- RMSE (original scale): ~$6.5k  
- MAE: ~$3.8k  

Given the wide range of used car prices, this is a solid result.

---

## Observations

- Model performs well for low–mid range cars  
- Slight underprediction for expensive/luxury cars  
- Price variability increases with higher-value vehicles  

This matches real-world behavior.

---

## Deployment

- Built using **Streamlit**  
- Deployed on **Render**  

The app allows users to input car details and get a predicted price instantly.

---

## Limitations

- Model depends on realistic input combinations  
- Sensitive to how categorical values are entered (e.g. model names)  
- Does not account for external factors (accidents, demand, etc.)

---

## Future Improvements

- Add input validation for better UX  
- Include prediction ranges (uncertainty)  
- Improve handling of rare/unseen categories  
- Add explainability (SHAP)  
- Build API version (FastAPI)

---

## Project Structure
used-car-price-prediction/

- |-- data/ (ignored in git)
- | |-- raw/
- | |-- interim/
- | |-- processed/
- |-- notebooks/
- | |-- 01_target_preprocessing.ipynb
- | |-- 02_feature_cleaning_preprocessing.ipynb
- | |-- 03_linear_models.ipynb
- | |-- 04_tree_models.ipynb
- | |-- 05_final_model_training_and_evaluation.ipynb
- |-- models/ (ignored in git)
- |-- requirements.txt
- |-- README.md

---

## Author

Harbaz Singh  