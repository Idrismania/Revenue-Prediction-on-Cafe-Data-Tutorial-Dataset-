# Coffee Shop Revenue Prediction

## Overview
This project analyzes daily revenue data from coffee shops to build a predictive model. By examining various operational factors like customer count, marketing spend, and staffing levels, the model helps predict daily revenue with high accuracy.

## Dataset
The dataset contains daily operational metrics from coffee shops, including:
- Number of customers per day
- Average order value
- Marketing spend
- Number of employees
- Operating hours
- Location foot traffic
- Daily revenue (target variable)

Data source: [Coffee Shop Daily Revenue Prediction Dataset](https://www.kaggle.com/datasets/himelsarder/coffee-shop-daily-revenue-prediction-dataset)

## Methodology

### Data Exploration
- Initial data visualization with pairplots to identify correlations
- Identified key relationships between customer numbers, order value, marketing spend and daily revenue

### Model Development
The project explores multiple modeling approaches with increasing complexity:
1. **Baseline Model**: Simple linear regression using only customer count
2. **Enhanced Baseline**: Linear regression with three key variables (customers, order value, marketing)
3. **Advanced Model**: Ridge regression with polynomial features after feature engineering and standardization

### Feature Engineering
Created meaningful derived features:
- Employees per customer ratio
- Marketing spend per customer
- Operating hours per employee
- Traffic per hour

## Results
The final Ridge regression model significantly outperforms the baseline approaches:

| Model | MSE | R² |
|-------|-----|---|
| Baseline (1-feature) | Higher | Lower |
| Enhanced Baseline | Moderate | Moderate |
| Ridge Regression | Lowest | Highest |

The final model demonstrates strong predictive capability by:
- Capturing non-linear relationships in the data
- Effectively utilizing engineered features
- Applying appropriate regularization to prevent overfitting

## Key Insights
- Customer count alone provides a moderate prediction of daily revenue
- Adding order value and marketing spend improves the prediction significantly
- Non-linear relationships and derived metrics like employees-per-customer further enhance prediction accuracy

## Usage
The notebook includes a complete inference pipeline for making predictions on new data:
```python
predicted_revenues = ridge_regression_inference(new_data, model_ridge, scaler, poly)
```

## Dependencies
- pandas
- matplotlib
- seaborn
- scikit-learn
- numpy
- pathlib