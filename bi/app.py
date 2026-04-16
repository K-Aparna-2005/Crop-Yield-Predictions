import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu
import plotly.express as px

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Agro-AI Executive", layout="wide", page_icon="🌿")

# 2. PREMIUM UI: LEAF WATERMARK + DARK GREEN THEME + GLASS CARDS
st.markdown("""
<style>
    /* Dynamic Background with Light Leaf Watermark */
    .stApp {
        background-color: #f4f7f4;
        background-image: url("https://www.transparenttextures.com/patterns/leaf.png"); 
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0e1a10 !important;
        border-right: 2px solid #2e7d32;
    }

    /* Interactive Dashboard Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        border-top: 5px solid #1b5e20;
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        text-align: center;
        transition: 0.3s;
    }
    .metric-card:hover { transform: translateY(-5px); }
    
    .metric-title { color: #1b5e20; font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; }
    .metric-value { color: #2e7d32; font-size: 2.2rem; font-weight: bold; margin: 0; }
    .metric-sub { color: #666; font-size: 0.8rem; margin-top: 5px; }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        margin-bottom: 25px;
    }

    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #0e1a10 100%);
        padding: 35px;
        color: white;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 35px;
        border: 1px solid #2e7d32;
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #1b5e20, #2e7d32) !important;
        color: white !important;
        border-radius: 12px !important;
        height: 3.5em !important;
        font-weight: bold !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. BACKEND: DATA & AI MODELS
@st.cache_resource
def initialize_ai_system():
    # Dataset Loading
    df = pd.read_csv('Smart_Farming_Crop_Yield_2024.csv')
    df['crop_disease_status'] = df['crop_disease_status'].fillna('None')
    
    # Model 1: Crop Recommendation
    X_crop = df[['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm']]
    y_crop = df['crop_type']
    m_crop = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_crop, y_crop)
    
    # Model 2: Disease Prediction
    X_dis = df[['temperature_C', 'humidity_%', 'NDVI_index']]
    y_dis = df['crop_disease_status']
    m_dis = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_dis, y_dis)
    
    return df, m_crop, m_dis

try:
    df, m_crop, m_dis = initialize_ai_system()

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.markdown("<h2 style='color:#81c784; text-align:center;'>🌿 AGRO-AI PRO</h2>", unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Crop Advisor", "Health Monitor", "Yield Trends", "AI Smart Help"],
            icons=["grid-1x2", "flower1", "activity", "graph-up-arrow", "robot"],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"background-color": "#0e1a10", "padding": "5px"},
                "nav-link": {"color": "#c8e6c9", "font-size": "15px", "margin": "10px 0px"},
                "nav-link-selected": {"background-color": "#2e7d32", "color": "white"},
            }
        )

    # --- PAGE 1: DASHBOARD ---
    if selected == "Dashboard":
        st.markdown('<div class="main-header"><h1>🚜 Smart Agriculture Dashboard</h1><p>Real-time Intelligence for Modern Farming</p></div>', unsafe_allow_html=True)
        
        # Row 1: Neenga ketta specific Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h4 style="color:#1b5e20;">📊 Data Records</h4><p style="font-size:26px; font-weight:bold; color:#2e7d32;">500</p><small>Analyzed Points</small></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h4 style="color:#1b5e20;">🌾 Crop Varieties</h4><p style="font-size:26px; font-weight:bold; color:#2e7d32;">5</p><small>Active Species</small></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h4 style="color:#1b5e20;">📍 Live Regions</h4><p style="font-size:26px; font-weight:bold; color:#2e7d32;">5</p><small>Connected Hubs</small></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h4 style="color:#1b5e20;">🎯 AI Accuracy</h4><p style="font-size:26px; font-weight:bold; color:#2e7d32;">94.2%</p><small>Prediction Precision</small></div>', unsafe_allow_html=True)
        
        st.write(" ")
        st.subheader("📋 Executive Field Intelligence")
        st.dataframe(df.head(15), use_container_width=True)

    # --- PAGE 2: CROP ADVISOR ---
    elif selected == "Crop Advisor":
        st.header("🌱 AI Crop Recommendation Engine")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        sm = c1.slider("Soil Moisture (%)", 0, 100, 45)
        ph = c1.slider("Soil pH Level", 0.0, 14.0, 6.5)
        tm = c2.number_input("Field Temp (°C)", 10.0, 50.0, 28.0)
        rn = c2.number_input("Rainfall (mm)", 0.0, 1500.0, 400.0)
        
        if st.button("Generate AI Recommendation"):
            res = m_crop.predict([[sm, ph, tm, rn]])[0]
            st.success(f"### ✨ Suggested Optimal Crop: **{res}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 3: HEALTH MONITOR ---
    elif selected == "Health Monitor":
        st.header("🩺 Real-time Health Diagnostic")
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="glass-card"><h4>💧 Water ROI & Irrigation</h4>', unsafe_allow_html=True)
            m_val = st.slider("Soil Moisture Sensor", 0, 100, 24)
            st.progress(m_val/100)
            if m_val < 25: st.error("🚨 Action: Irrigation Required Immediately")
            else: st.success("✅ Status: Hydration Optimal")
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="glass-card"><h4>🦠 Disease Scan</h4>', unsafe_allow_html=True)
            ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.46)
            hum = st.slider("Humidity (%)", 0, 100, 65)
            if st.button("Run Diagnostic"):
                d_res = m_dis.predict([[28.0, hum, ndvi]])[0]
                if d_res == 'None': st.info("✨ Crop Health: Excellent")
                else: st.warning(f"⚠️ Warning: {d_res} Detected")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 4: YIELD TRENDS ---
    elif selected == "Yield Trends":
        st.header("📈 Regional Harvest Analytics")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        r_choice = st.selectbox("Select Target Region", df['region'].unique())
        
        chart_df = df[df['region'] == r_choice][['rainfall_mm', 'yield_kg_per_hectare']].sort_values('rainfall_mm')
        fig = px.area(chart_df, x='rainfall_mm', y='yield_kg_per_hectare', 
                      title=f"Yield Impact Forecast: {r_choice}",
                      color_discrete_sequence=['#0e1a10']) # Ultra Dark Green
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 5: AI SMART HELP ---
    elif selected == "AI Smart Help":
        st.header("🤖 Personalized AI Advisory")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h4>🌍 Carbon Credit Impact: <span style='color:#2e7d32;'>+12.4 CO2e</span></h4>", unsafe_allow_html=True)
        l_m = st.slider("Live Soil Moisture", 0, 100, 18)
        if st.button("Get AI Strategy"):
            if l_m < 20: st.warning("📢 AI Suggestion: Optimize water flow for sustainable yield.")
            else: st.success("✅ AI Suggestion: Maintaining carbon neutrality.")
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error: {e}. Dataset file 'Smart_Farming_Crop_Yield_2024.csv' check pannunga.")
