import streamlit as st
import pandas as pd

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
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    text-align: center;
}

.metric-box {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("weather.csv")

# ==========================================
# PREPROCESSING
# ==========================================

label_encoder = LabelEncoder()

data['WeatherConditionEncoded'] = label_encoder.fit_transform(
    data['Weather Condition']
)

# ==========================================
# FEATURES
# ==========================================

X = data[['Humidity', 'Wind Speed', 'Pressure']]

# ==========================================
# TRAIN CLASSIFIER
# ==========================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(
    X,
    data['WeatherConditionEncoded']
)

# ==========================================
# TRAIN REGRESSOR
# ==========================================

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    X,
    data['Temperature']
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌦 Weather Dashboard")

st.sidebar.info("""
Machine Learning Project

Developer: Thanapal

Algorithms:
• Random Forest Classifier
• Random Forest Regressor

Platform:
• Python
• Streamlit
• Scikit-Learn
""")

# ==========================================
# HEADER
# ==========================================

st.title("🌦 Weather Prediction Dashboard")

st.markdown("""
### Predict Weather Conditions and Temperature using Machine Learning
""")

st.markdown("---")

# ==========================================
# METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset Records",
        len(data)
    )

with col2:
    st.metric(
        "Weather Classes",
        data['Weather Condition'].nunique()
    )

with col3:
    st.metric(
        "Average Temperature",
        round(data['Temperature'].mean(), 2)
    )

st.markdown("---")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📥 Enter Weather Parameters")

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

# ==========================================
# PREDICTION
# ==========================================

if st.button("🚀 Predict Weather"):

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
        f"🌤 Predicted Weather Condition: {weather_name[0]}"
    )

    st.success(
        f"🌡 Predicted Temperature: {round(temperature_prediction[0],2)} °C"
    )

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

    st.bar_chart(
        data['Weather Condition'].value_counts()
    )

with tab2:

    st.subheader("Temperature Distribution")

    st.line_chart(
        data['Temperature']
    )

with tab3:

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<center>

Developed by <b>Thanapal</b> 🚀

Weather Prediction using Machine Learning

</center>
""",
unsafe_allow_html=True
)
