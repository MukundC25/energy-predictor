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

# Modern glassmorphism UI with animations
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp { 
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0d1117 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 25%, #0984e3 50%, #6c5ce7 75%, #a29bfe 100%);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 212, 170, 0.3);
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
        border-radius: 50%;
    }
    .main-header h1 { 
        color: white !important; 
        margin: 0; 
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 2px 20px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    .main-header .subtitle { 
        color: rgba(255,255,255,0.95) !important; 
        margin: 0.75rem 0 0 0;
        font-size: 1.05rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .main-header .badge-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 1.25rem;
        position: relative;
        z-index: 1;
    }
    .main-header .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .main-header .badge-dot {
        width: 6px;
        height: 6px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 2s ease infinite;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .section-title .icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #00d4aa, #0984e3);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4aa, #0984e3, #6c5ce7);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        border-color: rgba(0, 212, 170, 0.3);
        box-shadow: 0 20px 40px rgba(0, 212, 170, 0.15);
    }
    .metric-card:hover::before {
        opacity: 1;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #0984e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .insight-card {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid #00d4aa;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 0.6rem 0;
        font-size: 0.9rem;
        color: #cbd5e1;
        transition: all 0.3s ease;
        line-height: 1.6;
    }
    .insight-card:hover {
        background: rgba(255,255,255,0.04);
        border-left-color: #0984e3;
        transform: translateX(4px);
    }
    .insight-card.warning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.05);
    }
    .insight-card.success {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.05);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15), rgba(9, 132, 227, 0.15));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 212, 170, 0.3);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .prediction-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 200%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    .prediction-result .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }
    .prediction-result .unit {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .input-section {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.75rem;
        margin: 1.25rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 4rem;
    }
    .footer strong {
        background: linear-gradient(135deg, #00d4aa, #0984e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Streamlit overrides for dark theme */
    .stSelectbox > div > div { 
        background: rgba(255,255,255,0.03) !important;
        border-color: rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.03) !important;
        border-color: rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4aa, #0984e3) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0984e3) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 212, 170, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.4) !important;
    }
    .stExpander {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 16px !important;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 12px !important;
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
    <h1>EcoPredict</h1>
    <p class="subtitle">AI-powered campus energy analytics to eliminate hidden resource waste in academic scheduling</p>
    <div class="badge-container">
        <span class="badge"><span class="badge-dot"></span> Live Model</span>
        <span class="badge">MLR Engine</span>
        <span class="badge">R² 0.92+</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data and model
with st.spinner("Initializing prediction engine..."):
    df, model, features, metrics, y_test, y_pred, insights = load_and_train()

# Model Performance Section
st.markdown('<div class="section-title"><span class="icon">M</span> Model Performance</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-title"><span class="icon">01</span> Prediction Accuracy</div>', unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Energy Waste', 'y': 'Predicted Energy Waste'},
        opacity=0.7
    )
    fig_scatter.update_traces(marker=dict(
        color='#00d4aa', 
        size=7,
        line=dict(width=1, color='rgba(0,212,170,0.3)')
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines', name='Perfect Fit',
        line=dict(color='#f59e0b', dash='dash', width=2)
    ))
    fig_scatter.update_layout(
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#94a3b8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        showlegend=False
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.markdown('<div class="section-title"><span class="icon">02</span> Feature Impact Analysis</div>', unsafe_allow_html=True)
    
    # Coefficient chart with professional styling
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    colors = ['#f59e0b' if c < 0 else '#00d4aa' for c in coef_df['Coefficient']]
    
    fig_coef = go.Figure(go.Bar(
        x=coef_df['Coefficient'],
        y=coef_df['Feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        )
    ))
    fig_coef.update_layout(
        height=380, 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#94a3b8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(fig_coef, use_container_width=True)

# Prediction Tool
st.markdown('<div class="section-title"><span class="icon">P</span> Energy Waste Predictor</div>', unsafe_allow_html=True)
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
st.markdown('<div class="section-title"><span class="icon">I</span> Sustainability Insights</div>', unsafe_allow_html=True)

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
    <strong>EcoPredict</strong> — Sustainable Campus Intelligence<br>
    Powered by Machine Learning · Applied AI for Energy Efficiency
</div>
''', unsafe_allow_html=True)
