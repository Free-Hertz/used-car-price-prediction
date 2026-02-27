# Used Car Price Prediction - End-to-End ML Pipeline

## Project Overview

This project builds a machine learning regression pipeline to recommend listing prices for used cars on an online marketplace.

The system predicts a fair market price using structured vehicle attributes such as year, mileage, manufacturer, model, condition, and region.

Beyond predictive accuracy, the goal was to simulate an industry-grade ML workflow including:

- Leakage prevention
- Feature engineering for high-cardinality variables
- Cross-validation-based model selection
- Strict holdout evaluation
- Error diagnostics and interpretation
- Reproducible sklearn pipeline design

Final dataset size after preprocessing: ~390,000 listings  
Final model: LightGBM Regressor  

---

## Business Problem

Online used-car marketplaces host highly variable listings where sellers often:

- Overprice vehicles -> slow sales
- Underprice vehicles -> revenue loss
- Misprice vehicles -> reduce buyer trust

A reliable price prediction model can:

- Recommend fair listing prices
- Flag suspicious listings
- Improve marketplace efficiency

---

## Target Variable

**Price (USD)** - Listed price of the used car.

To stabilize variance and reduce heteroscedasticity, the model was trained on:
log_price = log(price)


All final metrics are reported on both log scale and original dollar scale.

---

## Dataset & Preprocessing

Original dataset: ~426,000 rows  
Final modeling dataset: ~390,000 rows  

### Data Cleaning Steps

- Removed zero-priced listings (~5%)
- Trimmed extreme outliers (top 0.5 percentile)
- Applied log transformation to target
- Dropped leakage columns:
  - id, url, VIN, image_url
  - description
  - posting_date
  - latitude/longitude
  - county, state

---

## Feature Engineering Strategy

### Numerical Features
- year
- odometer

### High-Cardinality Categorical
- model (~27,000 unique values)
- region

Encoded using **Target Encoding** to preserve signal without dimensional explosion.

### Low-Cardinality Categorical
- manufacturer
- condition
- cylinders
- fuel
- title_status
- transmission
- drive
- size
- type
- paint_color

Encoded using **One-Hot Encoding**.

All transformations implemented using `ColumnTransformer` inside a sklearn `Pipeline`.

---

## Model Development

### Models Evaluated

- Linear Regression
- Ridge Regression
- Random Forest
- LightGBM

### Cross-Validation (Log RMSE)

| Model               | CV Log RMSE |
|---------------------|-------------|
| Linear Regression   | ~0.995      |
| Random Forest       | ~0.835      |
| LightGBM (baseline) | ~0.828      |
| Tuned LightGBM      | ~0.798      |

Tree-based models significantly outperformed linear models, confirming non-linear price relationships.

Final configuration:

- num_leaves = 63  
- learning_rate = 0.1  
- n_estimators = 400  
- max_depth = -1  

---

## Final Evaluation (Strict Holdout)

Train-test split:

- 80% training
- 20% test
- random_state = 42
- Test evaluated exactly once

### Test Performance

- Log RMSE: 0.7117
- RMSE (Original Scale): $6,525.81
- MAE (Original Scale): $3,880.82

### Interpretation

The model's average absolute error is approximately **$3,881 per vehicle**.

Given the wide variance in used car pricing, this represents strong predictive performance.

---

## Error Analysis

Diagnostic visualizations revealed:

- Strong calibration for low-to-mid price ranges
- Increasing variance for high-priced vehicles
- Mild underprediction bias in luxury segment
- No strong systematic overpricing bias

This heteroscedastic behavior is consistent with real-world marketplace dynamics.

---

## Feature Importance Insights

Top drivers of price prediction:

- Odometer
- Year
- Target-encoded model
- Manufacturer
- Region

Results align with expected real-world pricing factors.

---

## Engineering Design

- Full sklearn `Pipeline`
- Leakage-safe preprocessing
- Target encoding applied within cross-validation folds
- Reproducible training workflow
- Model artifact saved using `joblib`
- Binary artifacts excluded from Git

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

## Key Takeaways

- Log transformation significantly improved stability.
- Target Encoding effectively handled high-cardinality features.
- Tree models captured non-linear price structure.
- Strict holdout evaluation prevented leakage.
- Error diagnostics provided deeper model understanding beyond RMSE.

---

## Future Improvements

- SHAP-based explainability
- Luxury-segment specialized model
- Quantile regression for uncertainty estimation
- Streamlit deployment demo
- Modularizing preprocessing into `src/`

---

## Author

Harbaz Singh  