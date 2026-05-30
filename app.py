import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="WeatherAI",
    page_icon="🌦",
    layout="wide"
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
        #0f172a
    );
}

.hero {
    text-align:center;
    padding:40px 20px;
}

.hero h1{
    font-size:4rem;
    color:white;
    margin-bottom:10px;
}

.hero p{
    color:#cbd5e1;
    font-size:1.2rem;
}

.card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius:20px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.1);
}

.result-card{
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(15px);
    border-radius:25px;
    padding:30px;
    text-align:center;
    margin-top:20px;
}

.result-weather{
    font-size:50px;
    font-weight:bold;
    color:white;
}

.result-temp{
    font-size:42px;
    color:#38bdf8;
    font-weight:bold;
}

.small-text{
    color:#cbd5e1;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("weather.csv")

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
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🌦 WeatherAI</h1>
    <p>
        AI Powered Weather Forecasting Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT CARD
# ==========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Enter Weather Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    humidity = st.slider(
        "Humidity (%)",
        0,
        100,
        50
    )

with col2:
    windspeed = st.slider(
        "Wind Speed",
        0,
        50,
        10
    )

with col3:
    pressure = st.slider(
        "Pressure",
        900,
        1100,
        1000
    )

predict = st.button(
    "🚀 Generate Forecast",
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)

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
        "Rainy": "🌧️",
        "Cloudy": "☁️",
        "Stormy": "⛈️"
    }

    tips = {
        "Sunny":
            "Perfect weather for outdoor activities.",
        "Rainy":
            "Carry an umbrella before heading out.",
        "Cloudy":
            "Comfortable conditions expected today.",
        "Stormy":
            "Avoid outdoor travel if possible."
    }

    icon = icons.get(weather, "🌤")
    tip = tips.get(weather, "")

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-weather">
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

            <div class="small-text">
                {tip}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# EXTRA SECTION
# ==========================================

st.markdown("<br><br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.info("⚡ Instant AI Prediction")

with c2:
    st.info("🌎 Real-Time Style Dashboard")

with c3:
    st.info("🤖 Powered by Machine Learning")
