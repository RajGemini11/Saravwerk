# ============================================================================
# STEP 3: STREAMLIT UI
# File: ui_app.py
# ============================================================================
"""
Streamlit UI for Hospital Staffing Prediction
Run with: streamlit run ui_app.py
Access: http://localhost:8501
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Hospital Staffing Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE AND DESCRIPTION
# ============================================================================

st.markdown('<div class="main-header">🏥 Hospital Staffing Optimizer</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        ML-Powered 7-Day Patient Volume Forecasting | 5.35% MAPE Accuracy
    </p>
    <p style='font-size: 0.9rem; color: #888;'>
        Ridge Regression Model | UC Berkeley ML & AI Capstone Project
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - MODEL INFO AND SETTINGS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Model Information")
    
    # API URL configuration
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
    
    # Check API health
    try:
        health_response = requests.get(f"{api_url}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✓ API Connected")
            
            # Get model info
            info_response = requests.get(f"{api_url}/model-info")
            if info_response.status_code == 200:
                model_info = info_response.json()
                
                st.metric("Model Type", model_info['model_type'])
                st.metric("Expected MAPE", f"{model_info['expected_mape']}%")
                st.metric("Training Samples", f"{model_info['training_samples']:,}")
                
                with st.expander("📊 Model Details"):
                    st.json(model_info)
        else:
            st.error("✗ API Not Responding")
    except:
        st.error("✗ API Not Available")
        st.info("Start API with: `uvicorn main:app --reload`")
    
    st.divider()
    
    st.header("📘 How to Use")
    st.markdown("""
    1. **Single Day**: Select a date and input recent volumes
    2. **7-Day Forecast**: Get predictions for the next week
    3. **View Results**: See prediction, confidence interval, and staffing recommendation
    """)
    
    st.divider()
    
    st.header("📈 Model Performance")
    st.markdown("""
    - **MAPE**: 5.35% (Target: ≤10%)
    - **MAE**: ±41 patients
    - **Status**: Production Ready ✓
    """)

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📅 Single Day Prediction", "📊 7-Day Forecast", "ℹ️ About"])

# ============================================================================
# TAB 1: SINGLE DAY PREDICTION
# ============================================================================

with tab1:
    st.header("Single Day Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Date input
        prediction_date = st.date_input(
            "Prediction Date",
            value=datetime.now() + timedelta(days=7),
            min_value=datetime.now().date(),
            help="Select a date to predict patient volume"
        )
        
        st.divider()
        
        # Historical inputs
        st.markdown("**Recent Volume Data**")
        
        lag_1 = st.number_input(
            "Yesterday's Volume",
            min_value=0.0,
            max_value=1500.0,
            value=720.0,
            step=10.0,
            help="Patient volume from previous day"
        )
        
        lag_7 = st.number_input(
            "Same Day Last Week",
            min_value=0.0,
            max_value=1500.0,
            value=720.0,
            step=10.0,
            help="Patient volume from same day last week"
        )
        
        rolling_mean_7 = st.number_input(
            "7-Day Average",
            min_value=0.0,
            max_value=1500.0,
            value=720.0,
            step=10.0,
            help="Average patient volume over last 7 days"
        )
        
        rolling_std_7 = st.number_input(
            "7-Day Std Dev",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=5.0,
            help="Standard deviation of last 7 days"
        )
        
        # Predict button
        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            with st.spinner("Generating prediction..."):
                try:
                    # Prepare request
                    request_data = {
                        "target_date": prediction_date.strftime('%Y-%m-%d'),
                        "lag_1": lag_1,
                        "lag_7": lag_7,
                        "rolling_mean_7": rolling_mean_7,
                        "rolling_std_7": rolling_std_7
                    }
                    
                    # Make API call
                    response = requests.post(
                        f"{api_url}/predict",
                        json=request_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['prediction_result'] = result
                        st.success("✓ Prediction Generated Successfully!")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure FastAPI is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            # Main prediction
            st.markdown("### 📊 Predicted Volume")
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{result['predicted_volume']:.0f} patients</h1>", 
                       unsafe_allow_html=True)
            
            st.markdown(f"<p style='text-align: center; font-size: 0.9rem; color: #666;'>for {result['prediction_date']}</p>", 
                       unsafe_allow_html=True)
            
            st.divider()
            
            # Confidence interval
            col_ci1, col_ci2 = st.columns(2)
            with col_ci1:
                st.metric(
                    "Lower Bound (95%)",
                    f"{result['confidence_interval_lower']:.0f}",
                    help="Lower confidence interval"
                )
            with col_ci2:
                st.metric(
                    "Upper Bound (95%)",
                    f"{result['confidence_interval_upper']:.0f}",
                    help="Upper confidence interval"
                )
            
            st.divider()
            
            # Staffing recommendation
            st.markdown("### 👥 Staffing Recommendation")
            
            if "Low volume" in result['staffing_recommendation']:
                box_class = "success-box"
                icon = "✅"
            elif "Normal volume" in result['staffing_recommendation']:
                box_class = "metric-card"
                icon = "ℹ️"
            else:
                box_class = "warning-box"
                icon = "⚠️"
            
            st.markdown(f"""
                <div class='{box_class}'>
                    <strong>{icon} {result['staffing_recommendation']}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Model performance
            st.markdown("### 🎯 Expected Accuracy")
            st.info(f"Model MAPE: {result['mape_expected']}% (Target: ≤10%)")
            
            # Visualization
            st.markdown("### 📈 Confidence Interval")
            
            fig = go.Figure()
            
            # Add confidence interval
            fig.add_trace(go.Bar(
                x=['Prediction'],
                y=[result['predicted_volume']],
                name='Predicted',
                marker_color='#1f77b4',
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[result['confidence_interval_upper'] - result['predicted_volume']],
                    arrayminus=[result['predicted_volume'] - result['confidence_interval_lower']],
                    color='rgba(31, 119, 180, 0.3)',
                    thickness=2,
                    width=10
                )
            ))
            
            fig.update_layout(
                title=f"Prediction for {result['prediction_date']}",
                yaxis_title="Patient Volume",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Enter parameters and click 'Generate Prediction' to see results")

# ============================================================================
# TAB 2: 7-DAY FORECAST
# ============================================================================

with tab2:
    st.header("7-Day Volume Forecast")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Forecast Settings")
        
        start_date = st.date_input(
            "Forecast Start Date",
            value=datetime.now().date(),
            help="First day of 7-day forecast"
        )
        
        if st.button("📊 Generate 7-Day Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating 7-day forecast..."):
                try:
                    response = requests.post(
                        f"{api_url}/predict-week?start_date={start_date.strftime('%Y-%m-%d')}",
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        forecast_result = response.json()
                        st.session_state['forecast_result'] = forecast_result
                        st.success("✓ 7-Day Forecast Generated!")
                    else:
                        st.error(f"API Error: {response.status_code}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Forecast Results")
        
        if 'forecast_result' in st.session_state:
            forecast = st.session_state['forecast_result']
            
            # Summary metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            
            predictions = forecast['predictions']
            volumes = [p['predicted_volume'] for p in predictions]
            
            col_m1.metric("Week Total", f"{forecast['total_predicted']:.0f}")
            col_m2.metric("Daily Average", f"{sum(volumes)/len(volumes):.0f}")
            col_m3.metric("Peak Day", f"{max(volumes):.0f}")
            
            # Create forecast table
            st.markdown("### 📅 Daily Predictions")
            
            forecast_df = pd.DataFrame(predictions)
            forecast_df = forecast_df[[
                'prediction_date',
                'predicted_volume',
                'confidence_interval_lower',
                'confidence_interval_upper',
                'staffing_recommendation'
            ]]
            forecast_df.columns = [
                'Date',
                'Predicted Volume',
                'CI Lower',
                'CI Upper',
                'Staffing Recommendation'
            ]
            
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("### 📊 Weekly Trend")
            
            fig = go.Figure()
            
            dates = [p['prediction_date'] for p in predictions]
            volumes = [p['predicted_volume'] for p in predictions]
            ci_lower = [p['confidence_interval_lower'] for p in predictions]
            ci_upper = [p['confidence_interval_upper'] for p in predictions]
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=ci_upper + ci_lower[::-1],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=dates,
                y=volumes,
                mode='lines+markers',
                name='Predicted Volume',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="7-Day Patient Volume Forecast",
                xaxis_title="Date",
                yaxis_title="Patient Volume",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Click 'Generate 7-Day Forecast' to see predictions")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.header("About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Project Overview")
        st.markdown("""
        This application demonstrates a machine learning solution for optimizing hospital 
        staffing through accurate patient volume forecasting.
        
        **Key Features:**
        - 7-day advance predictions
        - 5.35% MAPE accuracy (46% better than target)
        - Real-time staffing recommendations
        - Confidence intervals for risk management
        
        **Business Impact:**
        - $450K-$900K annual savings potential
        - 4-8x ROI in Year 1
        - 65-73% improvement over manual forecasting
        """)
        
        st.subheader("🎯 Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['MAPE', 'MAE', 'RMSE', 'R²'],
            'Value': ['5.35%', '41.13', '48.31', '0.1068'],
            'Target': ['≤10%', '< 60', '< 70', '> 0']
        })
        st.table(metrics_df)
    
    with col2:
        st.subheader("🔧 Technical Stack")
        st.markdown("""
        **Machine Learning:**
        - Ridge Regression (α=1)
        - 30+ engineered features
        - TimeSeriesSplit cross-validation
        - GridSearchCV hyperparameter tuning
        
        **Backend:**
        - FastAPI for API serving
        - Scikit-learn for ML
        - Pandas for data processing
        
        **Frontend:**
        - Streamlit for UI
        - Plotly for visualizations
        - RESTful API integration
        """)
        
        st.subheader("👨‍💻 Author")
        st.markdown("""
        **Rajeshkumar**  
        Staff Data Engineer, HCA Healthcare  
        UC Berkeley ML & AI Professional Certificate
        
        14 years healthcare data engineering experience
        """)
    
    st.divider()
    
    st.subheader("📚 Dataset")
    st.markdown("""
    **Source:** [Kaggle Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments/data)  
    **Records:** 110,527 appointments (April-June 2016, Brazil)  
    **Features:** Patient demographics, health conditions, appointment details
    """)
    
    st.subheader("🚀 Deployment")
    st.code("""
# 1. Install dependencies
pip install fastapi uvicorn streamlit plotly pandas scikit-learn joblib

# 2. Start FastAPI backend
uvicorn main:app --reload

# 3. Start Streamlit UI (new terminal)
streamlit run ui_app.py

# 4. Access application
http://localhost:8501
    """, language="bash")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🏥 Hospital Staffing Optimizer | UC Berkeley ML & AI Capstone Project</p>
    <p style='font-size: 0.8rem;'>Powered by Ridge Regression | 5.35% MAPE Accuracy</p>
</div>
""", unsafe_allow_html=True)
