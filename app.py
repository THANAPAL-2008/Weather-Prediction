import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="WeatherAI",
    page_icon="🌤",
    layout="centered"
)

# =====================================
# HIDE STREAMLIT ELEMENTS
# =====================================

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

.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

h1 {
    text-align: center;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATASET
# =====================================

data = pd.read_csv("weather.csv")

# =====================================
# PREPROCESSING
# =====================================

label_encoder = LabelEncoder()

data["WeatherConditionEncoded"] = label_encoder.fit_transform(
    data["Weather Condition"]
)

# =====================================
# FEATURES
# =====================================

X = data[["Humidity", "Wind Speed", "Pressure"]]

# =====================================
# CLASSIFICATION MODEL
# =====================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X,
    data["WeatherConditionEncoded"]
)

# =====================================
# REGRESSION MODEL
# =====================================

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    X,
    data["Temperature"]
)

# =====================================
# HEADER
# =====================================

st.title("🌤 WeatherAI")

st.markdown(
"""
<div style='text-align:center;
font-size:22px;
margin-bottom:35px;'>

Predict weather instantly

</div>
""",
unsafe_allow_html=True
)

# =====================================
# INPUTS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    humidity = st.number_input(
        "Humidity",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

with col2:
    windspeed = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=10.0
    )

with col3:
    pressure = st.number_input(
        "Pressure",
        min_value=900.0,
        value=1000.0
    )

# =====================================
# PREDICTION BUTTON
# =====================================

predict = st.button(
    "Predict"
)

# =====================================
# PREDICTION
# =====================================

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

    icons = {
        "Sunny": "☀️",
        "Rainy": "🌧️",
        "Cloudy": "☁️",
        "Stormy": "⛈️"
    }

    weather = weather_name[0]

    icon = icons.get(
        weather,
        "🌤"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <h1 style='text-align:center'>
        {icon}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <h1 style='text-align:center'>
        {weather}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <h2 style='text-align:center'>
        {round(temperature_prediction[0],2)} °C
        </h2>
        """,
        unsafe_allow_html=True
    )
