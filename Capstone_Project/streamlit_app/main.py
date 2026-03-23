# ============================================================================
# STEP 2: FASTAPI BACKEND
# File: main.py
# ============================================================================
"""
FastAPI backend for serving hospital staffing predictions
Run with: uvicorn main:app --reload
Access: http://127.0.0.1:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

print("Loading hospital staffing model...")

try:
    model = joblib.load("hospital_staffing_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    metadata = joblib.load("model_metadata.pkl")
    print("✓ Model loaded successfully")
    print(f"  Model type: {metadata['model_type']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Expected MAPE: {metadata['expected_mape']}%")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class PredictionInput(BaseModel):
    """Input features for prediction"""
    target_date: str = Field(..., description="Date to predict (YYYY-MM-DD)")
    lag_1: Optional[float] = Field(None, description="Previous day volume")
    lag_7: Optional[float] = Field(None, description="Same day last week")
    rolling_mean_7: Optional[float] = Field(700.0, description="7-day average")
    rolling_std_7: Optional[float] = Field(50.0, description="7-day std dev")
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_date": "2024-06-15",
                "lag_1": 750.0,
                "lag_7": 720.0,
                "rolling_mean_7": 730.0,
                "rolling_std_7": 45.0
            }
        }

class PredictionOutput(BaseModel):
    """Prediction output"""
    predicted_volume: float
    prediction_date: str
    confidence_interval_lower: float
    confidence_interval_upper: float
    mape_expected: float
    staffing_recommendation: str

class ModelInfo(BaseModel):
    """Model metadata"""
    model_type: str
    alpha: float
    n_features: int
    training_samples: int
    training_date: str
    expected_mape: float

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Hospital Staffing Prediction API",
    description="ML-powered 7-day advance patient volume forecasting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def home():
    """Welcome endpoint"""
    return {
        "message": "Hospital Staffing Prediction API",
        "version": "1.0.0",
        "model": "Ridge Regression (α=1)",
        "mape": "5.35%",
        "endpoints": {
            "/predict": "POST - Make prediction",
            "/model-info": "GET - Model metadata",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info", response_model=ModelInfo)
def get_model_info():
    """Get model metadata"""
    return ModelInfo(**metadata)

@app.post("/predict", response_model=PredictionOutput)
def predict_volume(input_data: PredictionInput):
    """
    Predict patient volume for a given date
    
    Returns predicted volume with confidence interval and staffing recommendation
    """
    try:
        # Parse target date
        target_date = datetime.strptime(input_data.target_date, '%Y-%m-%d')
        
        # Create feature dataframe
        features_dict = {}
        
        # Temporal features
        features_dict['DayOfWeek'] = target_date.weekday()
        features_dict['DayOfMonth'] = target_date.day
        features_dict['WeekOfYear'] = target_date.isocalendar()[1]
        features_dict['Month'] = target_date.month
        features_dict['IsWeekend'] = 1 if target_date.weekday() >= 5 else 0
        
        # Day-of-week encoding
        for day in range(7):
            features_dict[f'Day_{day}'] = 1 if target_date.weekday() == day else 0
        
      # Lag features - UPDATED TO MATCH YOUR MODEL
        features_dict['Expected_Shows_Lag1'] = input_data.lag_1 or 720.0
        features_dict['Expected_Shows_Lag7'] = input_data.lag_7 or 720.0

      # Rolling statistics - UPDATED TO MATCH YOUR MODEL
        features_dict['Expected_Shows_Rolling7_Mean'] = input_data.rolling_mean_7 or 720.0
        features_dict['Expected_Shows_Rolling7_Std'] = input_data.rolling_std_7 or 50.0
        
        # Create DataFrame with correct column order
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[feature_columns]  # Ensure correct order
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Calculate confidence interval (±1.96 * RMSE from notebook: 48.31)
        rmse = 48.31
        ci_lower = prediction - (1.96 * rmse)
        ci_upper = prediction + (1.96 * rmse)
        
        # Staffing recommendation
        if prediction < 650:
            staff_rec = "Low volume expected - Reduce staff to baseline"
        elif prediction < 750:
            staff_rec = "Normal volume expected - Standard staffing"
        elif prediction < 850:
            staff_rec = "High volume expected - Add 1-2 extra staff"
        else:
            staff_rec = "Very high volume expected - Add 3+ extra staff"
        
        return PredictionOutput(
            predicted_volume=round(prediction, 1),
            prediction_date=input_data.target_date,
            confidence_interval_lower=round(ci_lower, 1),
            confidence_interval_upper=round(ci_upper, 1),
            mape_expected=5.35,
            staffing_recommendation=staff_rec
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict-week")
def predict_week(start_date: str):
    """
    Predict patient volumes for 7 days starting from start_date
    
    Returns list of predictions for the next week
    """
    try:
        predictions = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Default values for first prediction
        lag_1 = 720.0
        lag_7 = 720.0
        
        for day in range(7):
            pred_date = (current_date + timedelta(days=day)).strftime('%Y-%m-%d')
            
            input_data = PredictionInput(
                target_date=pred_date,
                lag_1=lag_1,
                lag_7=lag_7,
                rolling_mean_7=720.0,
                rolling_std_7=50.0
            )
            
            result = predict_volume(input_data)
            predictions.append(result.dict())
            
            # Update lags for next prediction
            lag_1 = result.predicted_volume
        
        return {
            "week_start": start_date,
            "predictions": predictions,
            "total_predicted": sum(p['predicted_volume'] for p in predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Week prediction error: {e}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("="*80)
    print("Starting Hospital Staffing Prediction API")
    print("="*80)
    print("URL: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs")
    print("="*80)
    uvicorn.run(app, host="127.0.0.1", port=8000)
