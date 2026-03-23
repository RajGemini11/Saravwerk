import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# THIS MUST BE FIRST!
st.set_page_config(
    page_title="Hospital Staffing Optimizer",
    page_icon="🏥",
    layout="wide"
)

# Load model directly (no API needed!)
@st.cache_resource
def load_model():
    model = joblib.load('hospital_staffing_model.pkl')
    features = joblib.load('feature_columns.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, features, metadata

model, feature_columns, metadata = load_model()

# Title
st.markdown('<div style="text-align: center;"><h1>🏥 Hospital Staffing Optimizer</h1></div>', 
            unsafe_allow_html=True)
st.markdown('<div style="text-align: center;"><p>ML-Powered 7-Day Patient Volume Forecasting | 4.79% MAPE Accuracy</p></div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Information")
    st.metric("Model Type", "Ridge Regression")
    st.metric("Expected MAPE", f"{metadata.get('expected_mape', 4.79):.2f}%")
    st.metric("Training Samples", metadata.get('training_samples', 19))
    
    st.divider()
    st.header("📘 How to Use")
    st.markdown("""
    1. Select prediction date
    2. Input recent volumes
    3. Generate prediction
    4. View staffing recommendation
    """)

# Main content
st.header("📅 Single Day Prediction")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Parameters")
    
    prediction_date = st.date_input(
        "Prediction Date",
        value=datetime.now(),
        help="Select date to predict"
    )
    
    st.divider()
    st.markdown("**Recent Volume Data**")
    
    lag_1 = st.number_input("Yesterday's Volume", value=720.0, step=10.0)
    lag_7 = st.number_input("Same Day Last Week", value=720.0, step=10.0)
    rolling_mean_7 = st.number_input("7-Day Average", value=720.0, step=10.0)
    rolling_std_7 = st.number_input("7-Day Std Dev", value=50.0, step=5.0)
    
    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        # Create features
        target_date = pd.to_datetime(prediction_date)
        
        features_dict = {
            'DayOfWeek': target_date.weekday(),
            'DayOfMonth': target_date.day,
            'WeekOfYear': target_date.isocalendar()[1],
            'Month': target_date.month,
            'IsWeekend': 1 if target_date.weekday() >= 5 else 0,
            'Expected_Shows_Lag1': lag_1,
            'Expected_Shows_Lag7': lag_7,
            'Expected_Shows_Rolling7_Mean': rolling_mean_7,
            'Expected_Shows_Rolling7_Std': rolling_std_7
        }
        
        # Day encoding
        for day in range(7):
            features_dict[f'Day_{day}'] = 1 if target_date.weekday() == day else 0
        
        # Create DataFrame
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[feature_columns]
        
        # Predict
        prediction = model.predict(features_df)[0]
        
        # Confidence interval
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
        
        # Store in session state
        st.session_state['prediction'] = {
            'volume': prediction,
            'date': prediction_date.strftime('%Y-%m-%d'),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'recommendation': staff_rec
        }
        
        st.success("✓ Prediction Generated Successfully!")

with col2:
    st.subheader("Prediction Results")
    
    if 'prediction' in st.session_state:
        result = st.session_state['prediction']
        
        st.markdown("### 📊 Predicted Volume")
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{result['volume']:.0f} patients</h1>", 
                   unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.2em;'>for {result['date']}</p>", 
                   unsafe_allow_html=True)
        
        st.divider()
        
        col_ci1, col_ci2 = st.columns(2)
        with col_ci1:
            st.markdown("**Lower Bound (95%)**")
            st.markdown(f"<h3 style='color: #ff7f0e;'>{result['ci_lower']:.0f}</h3>", unsafe_allow_html=True)
        with col_ci2:
            st.markdown("**Upper Bound (95%)**")
            st.markdown(f"<h3 style='color: #2ca02c;'>{result['ci_upper']:.0f}</h3>", unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 👥 Staffing Recommendation")
        st.info(result['recommendation'])
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Prediction'],
            y=[result['volume']],
            name='Predicted Volume',
            marker_color='#1f77b4',
            error_y=dict(
                type='data',
                symmetric=False,
                array=[result['ci_upper'] - result['volume']],
                arrayminus=[result['volume'] - result['ci_lower']],
                color='rgba(0,0,0,0.3)'
            )
        ))
        fig.update_layout(
            title=f"Prediction for {result['date']}", 
            yaxis_title="Patient Volume", 
            showlegend=False, 
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Enter parameters above and click 'Generate Prediction'")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🏥 Hospital Staffing Optimizer</strong></p>
    <p>UC Berkeley ML & AI Capstone Project</p>
    <p>Model Performance: 4.79% MAPE | Ridge Regression</p>
</div>
""", unsafe_allow_html=True)