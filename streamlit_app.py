"""
Sustainable Campus Energy Waste Predictor - Streamlit App
Single-page application for energy waste prediction and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from data_generation import generate_dataset
from regression_model import prepare_features, train_model, evaluate_model, get_insights

# Page config
st.set_page_config(
    page_title="Campus Energy Waste Predictor",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for sustainability theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2d5a27 0%, #4a7c43 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white !important; margin: 0; }
    .main-header p { color: #c8e6c9 !important; }
    .metric-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a7c43;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_train():
    """Load data and train model (cached)."""
    df = generate_dataset()
    X, y, features = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    insights = get_insights(model, features)
    return df, model, features, metrics, y_test, y_pred, insights


# Header
st.markdown("""
<div class="main-header">
    <h1>🌿 Campus Energy Waste Predictor</h1>
    <p>Predicting hidden resource waste from academic scheduling inefficiencies</p>
</div>
""", unsafe_allow_html=True)

# Load data and model
with st.spinner("Loading model..."):
    df, model, features, metrics, y_test, y_pred, insights = load_and_train()

# Layout: Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Performance")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{metrics['r2']:.3f}")
    m2.metric("RMSE", f"{metrics['rmse']:.2f}")
    m3.metric("MAE", f"{metrics['mae']:.2f}")
    
    # Predicted vs Actual
    st.subheader("Predicted vs Actual")
    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        opacity=0.5
    )
    fig_scatter.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines', name='Perfect Fit',
        line=dict(color='red', dash='dash')
    ))
    fig_scatter.update_layout(height=350)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("📈 Feature Importance")
    
    # Coefficient chart
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    fig_coef = px.bar(
        coef_df, x='Coefficient', y='Feature', orientation='h',
        color='Coefficient',
        color_continuous_scale=['#f44336', '#ffeb3b', '#4a7c43']
    )
    fig_coef.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_coef, use_container_width=True)

# Prediction Tool
st.markdown("---")
st.subheader("🔮 Predict Energy Waste")

p1, p2, p3, p4 = st.columns(4)
with p1:
    capacity = st.number_input("Room Capacity", 20, 100, 50)
    scheduled = st.number_input("Scheduled Students", 10, 100, 35)
with p2:
    attendance = st.number_input("Expected Attendance", 5, 100, 25)
    year = st.selectbox("Year of Study", [1, 2, 3, 4])
with p3:
    duration = st.selectbox("Duration (hours)", [1, 2, 3])
    time_slot = st.selectbox("Time Slot", ['morning', 'afternoon', 'evening'])
with p4:
    room_type = st.selectbox("Room Type", ['small_classroom', 'large_classroom', 'lab'])
    weather = st.slider("Weather Index", 0.2, 0.9, 0.5)

if st.button("Predict Energy Waste", type="primary"):
    # Prepare input
    util_ratio = attendance / capacity
    empty = capacity - attendance
    room_enc = {'small_classroom': 2, 'large_classroom': 1, 'lab': 0}[room_type]
    time_enc = {'morning': 1, 'afternoon': 0, 'evening': 2}[time_slot]
    day_enc = 2  # Wednesday (average)
    
    X_input = np.array([[capacity, scheduled, attendance, year, duration, 
                         weather, util_ratio, empty, room_enc, time_enc, day_enc]])
    
    prediction = model.predict(X_input)[0]
    
    st.success(f"**Predicted Energy Waste: {prediction:.2f} units**")
    
    if prediction > 8:
        st.warning("⚠️ High waste predicted. Consider a smaller room or earlier time slot.")
    elif prediction < 4:
        st.info("✅ Efficient scheduling. Room utilization is good.")

# Insights
st.markdown("---")
st.subheader("💡 Sustainability Insights")

for i, insight in enumerate(insights):
    st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)

# Dataset summary
with st.expander("📋 Dataset Summary"):
    st.write(f"**Total Records:** {len(df):,}")
    st.write(f"**Features:** {len(features)}")
    st.dataframe(df.describe().round(2))

# Footer
st.markdown("---")
st.caption("🌿 Applied AI for Sustainability | Campus Energy Waste Prediction System")
