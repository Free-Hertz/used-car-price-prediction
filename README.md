## Project Overview

This project focuses on building a regression model to recommend listing prices for used cars on an online marketplace.

The goal is to assist sellers by suggesting a reasonable listing price based on vehicle attributes, while also helping the platform identify suspiciously overpriced or underpriced listings.

## Business Problem

Online used-car marketplaces host millions of listings with highly variable pricing. Sellers often struggle to price their vehicles appropriately, leading to slow sales, mistrust, or potential fraud.

This project aims to predict a recommended listing price for a used car using historical listing data.

## Target Variable

- **Price**: The listed price of a used car (USD)

## Success Criteria

Model performance is primarily evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

These metrics ensure predictions remain consistently close to actual prices while minimizing large pricing errors that could negatively impact user trust or platform integrity.

### Data Cleaning (Completed)
- Cleaned and validated target variable (`price`)
- Removed invalid zero-priced listings (~5%)
- Trimmed extreme outliers using a conservative percentile cutoff
- Applied log transformation for modeling stability