import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
data = pd.read_csv("weather.csv")

# Encode Weather Condition
label_encoder = LabelEncoder()

data['WeatherConditionEncoded'] = label_encoder.fit_transform(
    data['Weather Condition']
)

# Features
X = data[['Humidity', 'Wind Speed', 'Pressure']]

# Classification Model
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X,
    data['WeatherConditionEncoded']
)

# Regression Model
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    X,
    data['Temperature']
)

# Website UI
st.title("🌦 Weather Prediction System")

st.write("Enter weather parameters below")

humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0
)

windspeed = st.number_input(
    "Wind Speed"
)

pressure = st.number_input(
    "Pressure"
)

if st.button("Predict"):

    custom_data = pd.DataFrame(
        [[humidity, windspeed, pressure]],
        columns=[
            'Humidity',
            'Wind Speed',
            'Pressure'
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

    st.success(
        f"Predicted Weather Condition: {weather_name[0]}"
    )

    st.success(
        f"Predicted Temperature: {round(temperature_prediction[0],2)} °C"
    )