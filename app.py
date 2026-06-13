import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as op

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS (Clean, Modern UI Elements)
# ==========================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        text-align: center;
        font-weight: 700;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# OPTIMIZED DATA & MODEL CACHING
# ==========================================
@st.cache_data
def load_data():
    # Production fallback/mock data helper if file missing
    try:
        df = pd.read_csv("weather.csv")
    except FileNotFoundError:
        # Fallback dataset if 'weather.csv' isn't local during a quick test
        np.random.seed(42)
        df = pd.DataFrame({
            'Humidity': np.random.uniform(30, 90, 200),
            'Wind Speed': np.random.uniform(0, 30, 200),
            'Pressure': np.random.uniform(980, 1030, 200),
            'Temperature': np.random.uniform(15, 38, 200),
            'Weather Condition': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Overcast'], 200)
        })
    
    le = LabelEncoder()
    df['WeatherConditionEncoded'] = le.fit_transform(df['Weather Condition'])
    return df, le

@st.cache_resource
def train_models(df):
    X = df[['Humidity', 'Wind Speed', 'Pressure']]
    
    # Train Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, df['WeatherConditionEncoded'])
    
    # Train Regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, df['Temperature'])
    
    return clf, reg

# Load data and models instantly from memory cache
data, label_encoder = load_data()
classifier, regressor = train_models(data)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("# 🌦 Weather Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🛠 Project Blueprint")
st.sidebar.info("""
**Developer:** Thanapal  
**Algorithms:** 
* Random Forest Classifier
* Random Forest Regressor  

**Stack:** Python, Streamlit, Scikit-Learn, Plotly
""")

st.sidebar.markdown("---")
st.sidebar.caption("Data Intelligence Dashboard © 2026")

# ==========================================
# HEADER
# ==========================================
st.title("🌦 Weather Prediction Dashboard")
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.2rem;'>Predict Weather Conditions and Temperature using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# METRICS (Enhanced with Custom Container UI)
# ==========================================
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Dataset Records</div>
        <div class="metric-val">{len(data)}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Weather Classes</div>
        <div class="metric-val">{data['Weather Condition'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Average Temperature</div>
        <div class="metric-val">{round(data['Temperature'].mean(), 1)} °C</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# INPUT & PREDICTION SECTION
# ==========================================
st.subheader("📥 Interactive Parameter Predictor")

# Putting Inputs on the left and Live Prediction on the right side
layout_col1, layout_col2 = st.columns([2, 1.3], gap="large")

with layout_col1:
    st.markdown("##### Adjust Parameters:")
    
    humidity = st.slider(
        "Humidity (%)",
        min_value=0.0, max_value=100.0, value=50.0, step=0.5
    )
    
    windspeed = st.slider(
        "Wind Speed (km/h)",
        min_value=0.0, max_value=100.0, value=12.0, step=0.5
    )
    
    pressure = st.slider(
        "Atmospheric Pressure (hPa)",
        min_value=900.0, max_value=1100.0, value=1013.0, step=0.1
    )

with layout_col2:
    st.markdown("##### Prediction Output:")
    
    # Process prediction live as user moves sliders (No submission button wall required!)
    custom_data = pd.DataFrame(
        [[humidity, windspeed, pressure]],
        columns=['Humidity', 'Wind Speed', 'Pressure']
    )
    
    weather_prediction = classifier.predict(custom_data)
    weather_name = label_encoder.inverse_transform(weather_prediction)[0]
    temperature_prediction = regressor.predict(custom_data)[0]
    
    # Visual cards for output results
    st.markdown(f"""
    <div style="background-color: #1E3A8A; padding: 18px; border-radius: 10px; margin-bottom: 12px; border-left: 6px solid #3B82F6;">
        <span style="color: #93C5FD; font-size: 0.85rem; text-transform: uppercase;">Condition Prediction</span>
        <h3 style="margin: 0; color: white; font-size: 1.6rem;">🌤 {weather_name}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color: #065F46; padding: 18px; border-radius: 10px; border-left: 6px solid #10B981;">
        <span style="color: #A7F3D0; font-size: 0.85rem; text-transform: uppercase;">Estimated Temperature</span>
        <h3 style="margin: 0; color: white; font-size: 1.6rem;">🌡 {round(temperature_prediction, 2)} °C</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# INTERACTIVE CHARTS (PLOTLY OVERHAUL)
# ==========================================
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Weather Distribution",
        "📈 Temperature Analytics",
        "📄 Dataset Exploration"
    ]
)

with tab1:
    st.subheader("Weather Condition Distribution")
    counts = data['Weather Condition'].value_counts().reset_index()
    counts.columns = ['Condition', 'Count']
    
    fig_bar = px.bar(
        counts, x='Condition', y='Count',
        color='Condition',
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        template="plotly_dark"
    )
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Temperature Trend Analysis")
    
    fig_line = px.line(
        data.reset_index(), x='index', y='Temperature',
        labels={'index': 'Data Row Index'},
        template="plotly_dark",
        color_discrete_sequence=['#F43F5E']
    )
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(
        data.drop(columns=['WeatherConditionEncoded'], errors='ignore'),
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<center style='color: #64748B;'>Developed with ❤️ by <b>Thanapal</b> | Powered by Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)
