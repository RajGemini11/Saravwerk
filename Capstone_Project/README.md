### Hospital Staff Optimization Using Machine Learning

Jupyter Notebook - https://github.com/RajGemini11/Saravwerk/blob/main/Capstone\_Project/Hospital\_Staff\_Optimization\_Model.ipynb

###### Research Question

Can machine learning models forecast daily expected patient volumes (accounting for no-shows) to optimize hospital staffing levels, achieving ≤10% MAPE for 7-day ahead predictions?

###### Dataset

Source: [Kaggle Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments/data)
Records: 110,527 appointments (April-June 2016, Brazil)
Features: Patient demographics, health conditions, appointment details, no-show status

###### Methodology

Applied CRISP-DM framework with comprehensive EDA, time series analysis, and feature engineering (30+ features including lag variables, rolling statistics, and temporal encodings). Developed baseline Ridge Regression model with proper time series validation.

###### Results

Baseline Model Performance (Ridge Regression):

* Test MAPE: 5.35%  (Target: ≤10%)
* MAE: 41.13 patients | RMSE: 48.31 | R²: 0.1068
* **Success:** Meets criteria with 45% margin, 65-73% better than manual forecasting

###### Key Findings

* Strong weekly seasonality (Monday peaks, weekend lows)
* Negative day-to-day autocorrelation (oscillation pattern)
* Lag-7 and rolling averages most predictive features
* Model enables reliable 7-day staffing decisions with ±50 patient buffer
* PACF and ACF evaluation with other models.

###### Next Steps (Module 24)

Compare advanced models (like Time Series SARIMAX,Random Forest, Linear Regression), hyperparameter tuning, ensemble methods, and deployment pipeline development.

