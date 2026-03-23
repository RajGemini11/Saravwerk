import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("="*60)
print("RETRAINING MODEL FOR STREAMLIT CLOUD COMPATIBILITY")
print("="*60)

# Use your cleaned daily_volumes (after removing bad day)
# Make sure you have df_simple from before

feature_columns = [
    'DayOfWeek', 'Month', 'DayOfMonth', 'WeekOfYear', 'IsWeekend',
    'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6',
    'Expected_Shows_Lag1', 'Expected_Shows_Lag7',
    'Expected_Shows_Rolling7_Mean', 'Expected_Shows_Rolling7_Std'
]

# Assuming you still have df_simple from earlier
X = df_simple[feature_columns]
y = df_simple['Expected_Shows']

print(f"Training samples: {len(X)}")
print(f"Target mean: {y.mean():.1f}")

# Train model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1, random_state=42))
])

model.fit(X, y)

# Test
test_pred = model.predict(X.tail(1))[0]
print(f"Test prediction: {test_pred:.1f}")

# Save with pickle protocol 4 for Python 3.8+ compatibility
print("\nSaving model files...")

# IMPORTANT: Use compress=3 and protocol=4 for compatibility
joblib.dump(model, 'hospital_staffing_model.pkl', compress=3, protocol=4)
joblib.dump(feature_columns, 'feature_columns.pkl', compress=3, protocol=4)

metadata = {
    'model_type': 'Ridge Regression',
    'alpha': 1,
    'n_features': len(feature_columns),
    'training_samples': len(X),
    'expected_mape': 4.79,
    'target_mean': float(y.mean())
}

joblib.dump(metadata, 'model_metadata.pkl', compress=3, protocol=4)