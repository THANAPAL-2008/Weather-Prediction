import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Weather & Temperature Prediction",
    layout="centered"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

.block-container {
    max-width: 850px;
    padding-top: 3rem;
}

.hero {
    text-align: center;
    margin-bottom: 40px;
}

.hero h1 {
    color: white;
    font-size: 3rem;
    margin-bottom: 10px;
}

.hero p {
    color: #cbd5e1;
    font-size: 1.15rem;
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.12);
}

.result-card {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin-top: 30px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
}

.result-weather {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.result-temp {
    font-size: 34px;
    font-weight: bold;
    color: #38bdf8;
}

.result-message {
    color: #cbd5e1;
    font-size: 18px;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATASET
# ==================================================

data = pd.read_csv("weather.csv")

# ==================================================
# PREPROCESSING
# ==================================================

label_encoder = LabelEncoder()

data["WeatherConditionEncoded"] = label_encoder.fit_transform(
    data["Weather Condition"]
)

X = data[["Humidity", "Wind Speed", "Pressure"]]

# ==================================================
# CLASSIFICATION MODEL
# ==================================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X,
    data["WeatherConditionEncoded"]
)

# ==================================================
# REGRESSION MODEL
# ==================================================

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    X,
    data["Temperature"]
)

# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🌦 Weather & Temperature Prediction</h1>
    <p>
        Predict weather conditions and temperature using machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# INPUT CARD
# ==================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)

windspeed = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    value=10.0,
    step=1.0
)

pressure = st.number_input(
    "Pressure (hPa)",
    min_value=900.0,
    max_value=1100.0,
    value=1000.0,
    step=1.0
)

predict = st.button(
    "Generate Forecast"
)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# PREDICTION
# ==================================================

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
        f"""
        <div class="result-card">

            <div style="font-size:70px;">
                {icon}
            </div>

            <div class="result-weather">
                {weather}
            </div>

            <br>

            <div class="result-temp">
                {round(temperature_prediction[0],2)} °C
            </div>

            <br>

            <div class="result-message">
                {message}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )
