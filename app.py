import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Weather Predictor",
    layout="centered"
)

# ==========================================
# MODERN MINIMALIST CLEAN CSS
# ==========================================
st.markdown("""
<style>
    /* Dark, professional background */
    .stApp {
        background-color: #0f172a;
    }

    /* Clean, Standard Title */
    .clean-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #f8fafc;
        margin-bottom: 5px;
    }
    
    .clean-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 35px;
    }

    /* Neat Framework for Input Fields */
    .input-box-frame {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 25px;
    }

    /* Standard Professional Button */
    div.stButton > button:first-child {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 1rem;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8 !important;
    }

    /* Simple, Flat Output Display Cards */
    .output-card-flat {
        background: #1e293b;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 12px;
    }
    
    .blue-side-border { border-left: 4px solid #3b82f6; }
    .green-side-border { border-left: 4px solid #10b981; }
    
    .card-lbl-text {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-val-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ML ENGINE CORE
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("weather.csv")
    except FileNotFoundError:
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
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, df['WeatherConditionEncoded'])
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, df['Temperature'])
    return clf, reg

data, label_encoder = load_data()
classifier, regressor = train_models(data)

# ==========================================
# SIMPLE HEADER
# ==========================================
st.markdown("<h1 class='clean-title'>Weather Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='clean-subtitle'>Enter details below to predict the weather condition</p>", unsafe_allow_html=True)

# ==========================================
# VERTICAL INPUT LAYOUT
# ==========================================
st.markdown('<div class="input-box-frame">', unsafe_allow_html=True)

humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=12.0)
pressure = st.number_input("Pressure (hPa)", min_value=900.0, value=1013.0)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# BUTTON & PREDICTIONS
# ==========================================
if st.button("Predict Weather", use_container_width=True):
    
    with st.spinner("Processing..."):
        time.sleep(0.2)
        
        custom_data = pd.DataFrame(
            [[humidity, windspeed, pressure]],
            columns=['Humidity', 'Wind Speed', 'Pressure']
        )
        
        weather_prediction = classifier.predict(custom_data)
        weather_name = label_encoder.inverse_transform(weather_prediction)[0]
        temperature_prediction = regressor.predict(custom_data)[0]
    
    # Minimalist Professional Output Cards
    st.markdown(f"""
    <div class="output-card-flat blue-side-border">
        <div class="card-lbl-text">Predicted Weather Condition</div>
        <div class="card-val-text">{weather_name}</div>
    </div>
    
    <div class="output-card-flat green-side-border">
        <div class="card-lbl-text">Predicted Temperature</div>
        <div class="card-val-text">{round(temperature_prediction, 1)} °C</div>
    </div>
    """, unsafe_allow_html=True)

# Remove standard platform sidebars, links, and headers
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
