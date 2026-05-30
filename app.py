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
    max-width:900px;
    padding-top:2rem;
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
    font-size:1.15rem;
    margin-bottom:35px;
}

.result-box{
    padding:35px;
    border-radius:20px;
    text-align:center;
    margin-top:25px;
}

div.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
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
# CLASSIFIER
# ==========================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X,
    data["WeatherConditionEncoded"]
)

# ==========================================
# REGRESSOR
# ==========================================

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
# INPUT SECTION
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

with col2:
    windspeed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=10.0
    )

with col3:
    pressure = st.number_input(
        "Pressure (hPa)",
        min_value=900.0,
        max_value=1100.0,
        value=1000.0
    )

predict = st.button(
    "Predict Weather"
)

# ==========================================
# PREDICTION
# ==========================================

if predict:

    with st.spinner("Generating forecast..."):

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

    colors = {
        "Sunny": "#f59e0b",
        "Cloudy": "#94a3b8",
        "Rainy": "#38bdf8",
        "Stormy": "#a855f7"
    }

    icon = icons.get(weather, "🌤")
    message = messages.get(weather, "")
    color = colors.get(weather, "#38bdf8")

    st.markdown(
        f"""
        <div class="result-box"
             style="
             background:rgba(255,255,255,0.08);
             border:1px solid rgba(255,255,255,0.15);
             ">

            <div style="font-size:70px;">
                {icon}
            </div>

            <h1 style="color:white;">
                {weather}
            </h1>

            <h2 style="color:{color};
                       font-size:42px;">
                {round(temperature_prediction[0],2)} °C
            </h2>

            <p style="
               color:#cbd5e1;
               font-size:18px;">
               {message}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )
