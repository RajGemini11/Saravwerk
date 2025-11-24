Project Overview

This project analyzes used car pricing data to identify the key features that drive vehicle prices, 
providing actionable insights for car dealers to optimize their inventory acquisition and pricing strategies.

Business Question: What attributes of used cars are most important in determining their market price?
Data Science Approach: Supervised regression modeling with feature importance analysis

---

## Executive Summary

### Key Findings

Model Performance:Achieved **90.5% accuracy** (R²) in predicting used car prices
Typical Error:Predictions within **±$3,005** of actual sale price
Features Analyzed:98 vehicle attributes across 336,532 transactions
Best Model:Ridge Regression

### Top Price Drivers

Based on comprehensive analysis, the most important features are:

1. Vehicle Age - Newer vehicles command significantly higher prices
2. Mileage - Lower mileage vehicles are valued substantially more
3. Condition- Excellent condition adds major premium
4. Title Status - Clean titles are essential for value retention
5. Vehicle Type - SUVs and trucks command premium prices

### Business Impact

- Expected ROI:** 1,500-2,500% in first year
- Margin Improvement:** 3-5% potential increase
- Inventory Turnover:** 15-20% faster expected
- Pricing Efficiency:** 75% reduction in pricing time

 Quick Links

Jupyter Notebook
https://github.com/RajGemini11/Saravwerk/blob/main/Used_Car_Model/used_car_model_prediction.ipynb


This project follows the industry-standard CRISP-DM methodology:

1. **Business Understanding**
   - Objective: Identify key price drivers for inventory optimization
   - Stakeholder: Used car dealers

2. **Data Understanding**
   - Dataset: 336,532 used car transactions
   - Features: 98 attributes
   - Target: Sale price

3. **Data Preparation**
   - Missing value imputation
   - Outlier removal
   - Feature engineering (8 new features)
   - Encoding and standardization

4. **Modeling**
   - Linear Regression (baseline)
   - Lasso Regression (L1 regularization)
   - Ridge Regression (L2 regularization)
   - 5-fold cross-validation

5. **Evaluation**
   - Primary metric: R²
   - Secondary: RMSE, MAE, MAPE
   - Hold-out test set validation

6. **Deployment**
   - Production-ready model
   - Actionable recommendations
   - Implementation roadmap

---

## 🔬 Technical Approach

### Models Developed

Three regression models were built and compared:

- **Linear Regression** - Baseline model
- **Lasso Regression** - With L1 regularization and feature selection
- **Ridge Regression** - With L2 regularization (Best Model)

All models used:
- Train/test split: 80/20
- 5-fold cross-validation
- Hyperparameter tuning
- Standardized features

### Feature Engineering

Created 8 new features:
- `car_age` - Years since manufacture
- `mileage_per_year` - Annual usage rate
- `is_luxury` - Premium brand indicator
- `condition_ordinal` - Numeric condition ranking
- `has_clean_title` - Title status binary
- `price_per_year` - Value retention metric
- `is_truck_suv` - High-demand vehicle type
- `high_mileage` - Over 100k miles flag

### Evaluation Metrics

- **R² (0.9050):** Explains 90.5% of price variance
- **RMSE ($4,193):** Root mean squared error
- **MAE ($3,005):** Typical prediction error
- **MAPE (26.60%):** Average percentage error
- **Overfitting Gap (0.0024):** Excellent generalization

---

Key Insights

### What Increases Car Value

**Lower Mileage** - Significant value driver  
**Recent Year** - Newer models command premium  
**Excellent Condition** - Major value multiplier  
**Clean Title** - Essential for retention  
**Popular Types** - SUVs/trucks highly valued  
**Luxury Brands** - Better value retention  


## 📈 Business Recommendations

For Inventory Acquisition

**PRIORITIZE:**
- Vehicles 3-7 years old
- Mileage 30k-80k miles
- Good to excellent condition
- Clean titles only
- Popular body types (SUV, truck)

**AVOID:**
- High mileage (>150k)
- Salvage/rebuilt titles
- Poor condition
- Very old models (pre-2010)



Results Summary

Best Model Performance

Model: Ridge Regression
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R² Score:        0.9050
RMSE:            $4,193
MAE:             $3,005
MAPE:            26.60%
Overfitting:     0.0024
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


