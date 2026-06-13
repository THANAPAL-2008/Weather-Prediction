import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    layout="wide"
)

# ==========================================
# DATA & MODEL CACHING
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
# SIDEBAR
# ==========================================
st.sidebar.title("Dashboard Options")
st.sidebar.markdown("---")
st.sidebar.write("Use the controls on the main page to input predictive features.")

# ==========================================
# MAIN HEADER
# ==========================================
st.title("Weather Prediction Dashboard")
st.markdown("Predict predictive outcomes using trained Machine Learning models.")
st.markdown("---")

# ==========================================
# METRICS
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(data))

with col2:
    st.metric("Unique Weather Classes", data['Weather Condition'].nunique())

with col3:
    st.metric("Mean Temperature", f"{round(data['Temperature'].mean(), 2)} °C")

st.markdown("---")

# ==========================================
# INPUT SECTION
# ==========================================
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0, value=50.0
    )

with col2:
    windspeed = st.number_input(
        "Wind Speed",
        min_value=0.0, value=10.0
    )

with col3:
    pressure = st.number_input(
        "Atmospheric Pressure",
        min_value=900.0, value=1000.0
    )

# ==========================================
# PREDICTION LOGIC
# ==========================================
custom_data = pd.DataFrame(
    [[humidity, windspeed, pressure]],
    columns=['Humidity', 'Wind Speed', 'Pressure']
)

weather_prediction = classifier.predict(custom_data)
weather_name = label_encoder.inverse_transform(weather_prediction)[0]
temperature_prediction = regressor.predict(custom_data)[0]

# ==========================================
# PREDICTION OUTPUT (Minimalist Cards)
# ==========================================
st.markdown("---")
st.subheader("Model Predictions")

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.info(f"Predicted Condition: **{weather_name}**")

with res_col2:
    st.info(f"Predicted Temperature: **{round(temperature_prediction, 2)} °C**")

st.markdown("---")

# ==========================================
# DATASET PREVIEW
# ==========================================
st.subheader("Dataset Summary")
st.dataframe(
    data.drop(columns=['WeatherConditionEncoded'], errors='ignore').head(10),
    use_container_width=True
)
