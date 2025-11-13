import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Will it rain tomorrow? ☔",
    page_icon="☔",
    layout="centered",
)

st.title("Will it rain tomorrow? ☔")
st.markdown(
    """
Прогноз ймовірності дощу завтра на основі моделі логістичної регресії, 
навченої на даних погодних спостережень по Австралії.
"""
)

# =========================
# 1. Завантаження пайплайна
# =========================
@st.cache_resource
def load_pipeline():
    path = Path("aussie_rain_pipeline.joblib")
    if not path.exists():
        st.error("Файл aussie_rain_pipeline.joblib не знайдено у корені. "
                 "Скопіюй його в ту ж папку, де лежить app.py.")
        st.stop()
    pipe = joblib.load(path)
    return pipe

model_pipeline = load_pipeline()

# Список колонок, які ти використовувала як input_cols
INPUT_COLS = [
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
]

# Значення за замовчуванням / списки категорій
# (можеш розширити або оновити, але handle_unknown='ignore', тож буде ок)
LOCATIONS = [
    "Albury", "Sydney", "Melbourne", "Canberra", "Brisbane",
    "Adelaide", "Perth", "Hobart", "Darwin"
]

WIND_DIRS = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW"
]

RAIN_TODAY_OPTIONS = ["No", "Yes"]

# =========================
# 2. Форма вводу даних
# =========================
st.subheader("Введи сьогоднішні погодні умови")

with st.form("weather_form"):
    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location", LOCATIONS)
        min_temp = st.number_input("MinTemp (°C)", value=10.0, step=0.5)
        max_temp = st.number_input("MaxTemp (°C)", value=20.0, step=0.5)
        rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)
        evaporation = st.number_input("Evaporation (mm)", value=5.0, step=0.1)
        sunshine = st.number_input("Sunshine (hours)", value=7.0, step=0.1)
        wind_gust_dir = st.selectbox("WindGustDir", WIND_DIRS)
        wind_gust_speed = st.number_input("WindGustSpeed (km/h)", value=35.0, step=1.0)
        wind_dir_9am = st.selectbox("WindDir9am", WIND_DIRS)
        wind_dir_3pm = st.selectbox("WindDir3pm", WIND_DIRS)

    with col2:
        wind_speed_9am = st.number_input("WindSpeed9am (km/h)", value=15.0, step=1.0)
        wind_speed_3pm = st.number_input("WindSpeed3pm (km/h)", value=20.0, step=1.0)
        humidity_9am = st.number_input("Humidity9am (%)", value=70.0, step=1.0)
        humidity_3pm = st.number_input("Humidity3pm (%)", value=50.0, step=1.0)
        pressure_9am = st.number_input("Pressure9am (hPa)", value=1015.0, step=0.5)
        pressure_3pm = st.number_input("Pressure3pm (hPa)", value=1012.0, step=0.5)
        cloud_9am = st.slider("Cloud9am (oktas 0–9)", min_value=0, max_value=9, value=4)
        cloud_3pm = st.slider("Cloud3pm (oktas 0–9)", min_value=0, max_value=9, value=4)
        temp_9am = st.number_input("Temp9am (°C)", value=16.0, step=0.5)
        temp_3pm = st.number_input("Temp3pm (°C)", value=21.0, step=0.5)
        rain_today = st.selectbox("RainToday", RAIN_TODAY_OPTIONS)

    submitted = st.form_submit_button("Зробити прогноз ☁️")

# =========================
# 3. Прогноз
# =========================
if submitted:
    # Один рядок з усіма колонками, які ти подавала при тренуванні
    raw_input = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
    }

    # Переконаємось, що порядок колонок такий самий, як INPUT_COLS
    input_df = pd.DataFrame([[raw_input[col] for col in INPUT_COLS]], columns=INPUT_COLS)

    try:
        proba = model_pipeline.predict_proba(input_df)[0, 1]  # ймовірність класу "Yes"
        pred_label = model_pipeline.predict(input_df)[0]      # "Yes" / "No"

        st.subheader("Результат прогнозу")

        if pred_label == "Yes":
            st.markdown("🌧 **Завтра очікується дощ.**")
        else:
            st.markdown("🌤 **Завтра дощу не очікується.**")

        st.metric("Ймовірність дощу завтра", f"{proba * 100:.1f} %")

        with st.expander("Деталі введених даних"):
            st.write(input_df)

    except Exception as e:
        st.error(f"Щось пішло не так під час прогнозу: {e}")
        st.info(
            "Перевір, що назви колонок у INPUT_COLS збігаються з input_cols, "
            "які ти використовувала в ноутбуці."
        )