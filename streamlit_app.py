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
    page_title="EcoPredict | Campus Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS with industry-standard styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 50%, #3d7a4f 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .main-header h1 { 
        color: white !important; 
        margin: 0; 
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header .subtitle { 
        color: rgba(255,255,255,0.8) !important; 
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 400;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a472a;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8f5e9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title .icon {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #2d5a3d, #4a7c5a);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.8rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a472a;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .insight-card {
        background: linear-gradient(to right, #f8fdf9, #ffffff);
        border: 1px solid #c8e6c9;
        border-left: 4px solid #2d5a3d;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        color: #2c3e2e;
    }
    .insight-card.warning {
        background: linear-gradient(to right, #fffbf5, #ffffff);
        border-color: #ffe0b2;
        border-left-color: #f57c00;
    }
    .insight-card.success {
        background: linear-gradient(to right, #f1f8e9, #ffffff);
        border-color: #c5e1a5;
        border-left-color: #558b2f;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #1a472a, #2d5a3d);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-result .value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .prediction-result .unit {
        font-size: 1rem;
        opacity: 0.8;
    }
    
    .input-section {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Override Streamlit defaults */
    .stMetric { background: transparent !important; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
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
    <h1>EcoPredict — Campus Energy Analytics</h1>
    <p class="subtitle">Machine learning system for identifying hidden resource waste in academic scheduling</p>
    <span class="badge">Multiple Linear Regression • R² 0.92+</span>
</div>
""", unsafe_allow_html=True)

# Load data and model
with st.spinner("Initializing prediction engine..."):
    df, model, features, metrics, y_test, y_pred, insights = load_and_train()

# Model Performance Section
st.markdown('<div class="section-title"><span class="icon">◆</span> Model Performance Metrics</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'''
    <div class="metric-card">
        <div class="value">{metrics['r2']:.3f}</div>
        <div class="label">R² Score</div>
    </div>
    ''', unsafe_allow_html=True)
with m2:
    st.markdown(f'''
    <div class="metric-card">
        <div class="value">{metrics['rmse']:.2f}</div>
        <div class="label">RMSE</div>
    </div>
    ''', unsafe_allow_html=True)
with m3:
    st.markdown(f'''
    <div class="metric-card">
        <div class="value">{metrics['mae']:.2f}</div>
        <div class="label">MAE</div>
    </div>
    ''', unsafe_allow_html=True)
with m4:
    st.markdown(f'''
    <div class="metric-card">
        <div class="value">{len(df):,}</div>
        <div class="label">Training Samples</div>
    </div>
    ''', unsafe_allow_html=True)

# Two column layout for charts
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-title"><span class="icon">◇</span> Prediction Accuracy</div>', unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Energy Waste', 'y': 'Predicted Energy Waste'},
        opacity=0.6
    )
    fig_scatter.update_traces(marker=dict(color='#2e7d32', size=6))
    fig_scatter.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines', name='Perfect Fit',
        line=dict(color='#c62828', dash='dash', width=2)
    ))
    fig_scatter.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.markdown('<div class="section-title"><span class="icon">◆</span> Feature Impact Analysis</div>', unsafe_allow_html=True)
    
    # Coefficient chart with professional styling
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    fig_coef = px.bar(
        coef_df, x='Coefficient', y='Feature', orientation='h',
        color='Coefficient',
        color_continuous_scale=['#c62828', '#fdd835', '#2e7d32']
    )
    fig_coef.update_layout(
        height=350, 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter')
    )
    st.plotly_chart(fig_coef, use_container_width=True)

# Prediction Tool
st.markdown('<div class="section-title"><span class="icon">▶</span> Energy Waste Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="input-section">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Calculate Energy Waste", type="primary", use_container_width=True):
    # Prepare input
    util_ratio = attendance / capacity
    empty = capacity - attendance
    room_enc = {'small_classroom': 2, 'large_classroom': 1, 'lab': 0}[room_type]
    time_enc = {'morning': 1, 'afternoon': 0, 'evening': 2}[time_slot]
    day_enc = 2  # Wednesday (average)
    
    X_input = np.array([[capacity, scheduled, attendance, year, duration, 
                         weather, util_ratio, empty, room_enc, time_enc, day_enc]])
    
    prediction = model.predict(X_input)[0]
    utilization = (attendance / capacity) * 100
    
    # Display prediction with professional styling
    st.markdown(f'''
    <div class="prediction-result">
        <div class="value">{prediction:.2f}</div>
        <div class="unit">Estimated Energy Waste Units</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Contextual feedback
    if prediction > 8:
        st.markdown(f'<div class="insight-card warning">High waste detected ({utilization:.0f}% utilization). Consider a smaller room or morning schedule.</div>', unsafe_allow_html=True)
    elif prediction < 4:
        st.markdown(f'<div class="insight-card success">Efficient scheduling. Room utilization at {utilization:.0f}% is optimal.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="insight-card">Moderate waste level. Room utilization: {utilization:.0f}%</div>', unsafe_allow_html=True)

# Insights Section
st.markdown('<div class="section-title"><span class="icon">◈</span> Key Sustainability Insights</div>', unsafe_allow_html=True)

col_ins1, col_ins2 = st.columns(2)
for i, insight in enumerate(insights):
    with col_ins1 if i % 2 == 0 else col_ins2:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

# Dataset summary
with st.expander("View Dataset Statistics"):
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Features:** {len(features)}")
    with c2:
        st.write(f"**Mean Waste:** {df['energy_waste'].mean():.2f} units")
        st.write(f"**Waste Range:** {df['energy_waste'].min():.2f} - {df['energy_waste'].max():.2f}")
    st.dataframe(df.describe().round(2), use_container_width=True)

# Footer
st.markdown('''
<div class="footer">
    <strong>EcoPredict</strong> — Campus Energy Waste Prediction System<br>
    Applied Artificial Intelligence for Sustainability
</div>
''', unsafe_allow_html=True)
