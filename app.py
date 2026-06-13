import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌦",
    layout="wide"
)

# ==========================================
# DATA & MODEL CACHING (Keeps it lightning fast)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("weather.csv")
    except FileNotFoundError:
        # Emergency local fallback data
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
st.sidebar.title("🌦 Weather Dashboard")

st.sidebar.info("""
Machine Learning Project

**Developer:** Thanapal

**Algorithms:**
• Random Forest Classifier
• Random Forest Regressor

**Platform:**
• Python
• Streamlit
• Scikit-Learn
""")

# ==========================================
# HEADER
# ==========================================
st.title("🌦 Weather Prediction Dashboard")
st.markdown("### Predict Weather Conditions and Temperature using Machine Learning")
st.markdown("---")

# ==========================================
# METRICS (Standard Minimalist Style)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", len(data))

with col2:
    st.metric("Weather Classes", data['Weather Condition'].nunique())

with col3:
    st.metric("Average Temperature", f"{round(data['Temperature'].mean(), 2)} °C")

st.markdown("---")

# ==========================================
# INPUT SECTION (Classic Number Inputs)
# ==========================================
st.subheader("📥 Enter Weather Parameters")

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
        "Pressure",
        min_value=900.0, value=1000.0
    )

# ==========================================
# PREDICTION OUTPUT (Clean, Non-Flashy Banners)
# ==========================================
custom_data = pd.DataFrame(
    [[humidity, windspeed, pressure]],
    columns=['Humidity', 'Wind Speed', 'Pressure']
)

weather_prediction = classifier.predict(custom_data)
weather_name = label_encoder.inverse_transform(weather_prediction)[0]
temperature_prediction = regressor.predict(custom_data)[0]

st.markdown("<br>", unsafe_allow_html=True)
st.success(f"🌤 Predicted Weather Condition: **{weather_name}**")
st.success(f"🌡 Predicted Temperature: **{round(temperature_prediction, 2)} °C**")

st.markdown("---")

# ==========================================
# CHARTS
# ==========================================
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Weather Distribution",
        "📈 Temperature Distribution",
        "📄 Dataset Preview"
    ]
)

with tab1:
    st.subheader("Weather Condition Distribution")
    counts = data['Weather Condition'].value_counts().reset_index()
    counts.columns = ['Condition', 'Count']
    
    fig_bar = px.bar(counts, x='Condition', y='Count', template="plotly_dark")
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Temperature Distribution")
    fig_line = px.line(data.reset_index(), x='index', y='Temperature', template="plotly_dark")
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(
        data.drop(columns=['WeatherConditionEncoded'], errors='ignore').head(20),
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Thanapal</b> 🚀<br>Weather Prediction using Machine Learning</center>",
    unsafe_allow_html=True
)
