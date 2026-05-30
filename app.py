import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Weather & Temperature Prediction",
    page_icon="🌦",
    layout="centered"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

.block-container{
    padding-top:3rem;
    max-width:850px;
}

.main-title{
    text-align:center;
    color:white;
    font-size:3rem;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:1.2rem;
    margin-bottom:30px;
}

.result-box{
    background: rgba(255,255,255,0.08);
    padding:30px;
    border-radius:20px;
    text-align:center;
    margin-top:25px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("weather.csv")

# ==========================================
# PREPROCESSING
# ==========================================

label_encoder = LabelEncoder()

data["WeatherConditionEncoded"] = label_encoder.fit_transform(
    data["Weather Condition"]
)

X = data[["Humidity", "Wind Speed", "Pressure"]]

# ==========================================
# MODELS
# ==========================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X,
    data["WeatherConditionEncoded"]
)

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    X,
    data["Temperature"]
)

# ==========================================
# HEADER
# ==========================================

st.markdown(
"""
<div class="main-title">
🌦 Weather & Temperature Prediction
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">
AI-Powered Weather Forecasting System
</div>
""",
unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# INPUTS
# ==========================================

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

windspeed = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    value=10.0
)

pressure = st.number_input(
    "Pressure (hPa)",
    min_value=900.0,
    max_value=1100.0,
    value=1000.0
)

predict = st.button(
    "Generate Forecast",
    use_container_width=True
)

# ==========================================
# PREDICTION
# ==========================================

if predict:

    custom_data = pd.DataFrame(
        [[humidity, windspeed, pressure]],
        columns=[
            "Humidity",
            "Wind Speed",
            "Pressure"
        ]
    )

    weather_prediction = classifier.predict(
        custom_data
    )

    weather_name = label_encoder.inverse_transform(
        weather_prediction
    )

    temperature_prediction = regressor.predict(
        custom_data
    )

    weather = weather_name[0]

    icons = {
        "Sunny": "☀️",
        "Cloudy": "☁️",
        "Rainy": "🌧️",
        "Stormy": "⛈️"
    }

    messages = {
        "Sunny": "Perfect weather for outdoor activities.",
        "Cloudy": "Comfortable conditions expected.",
        "Rainy": "Carry an umbrella before heading out.",
        "Stormy": "Avoid outdoor travel if possible."
    }

    icon = icons.get(weather, "🌤")
    message = messages.get(weather, "")

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h1 style='text-align:center'>{icon}</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h2 style='text-align:center'>{weather}</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h1 style='text-align:center;color:#38bdf8'>{round(temperature_prediction[0],2)} °C</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='text-align:center'>{message}</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
