import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt

# Пути к файлам модели и масштабатора
model_path = r"C:\Диск D\Python\Model\New_Model\yield_model.h5"
scaler_path = r"C:\Диск D\Python\Model\New_Model\scaler.save"

# Загрузка модели
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
else:
    st.error("❌ Модель не найдена. Проверьте путь.")
    st.stop()

# Загрузка масштабатора
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("❌ Масштабатор не найден. Проверьте путь.")
    st.stop()

st.set_page_config(page_title="Прогноз урожайности", layout="centered")
st.title("🌾 Прогноз урожайности (т/га) по агропараметрам")

# --- Выбор поля (для удобства анализа) ---
st.subheader("1. Выбор поля")
field_id = st.selectbox("Выберите номер поля", options=[1, 2, 3])

# --- Ввод параметров ---
st.subheader("2. Введите агрометеорологические и агрохимические данные")

temp_min = st.slider("Минимальная температура (°C)", -5.0, 5.0, 0.0)
temp_avg = st.slider("Средняя температура (°C)", 5.0, 20.0, 10.0)
temp_max = st.slider("Максимальная температура (°C)", 20.0, 40.0, 30.0)
rainfall = st.slider("Сумма осадков за сезон (мм)", 0, 600, 300)
ph = st.slider("pH почвы", 4.0, 9.0, 6.0)
moisture = st.slider("Влажность почвы (%)", 0, 60, 30)
st.header("3. Выберите соотношения удобрения: Диаммофоска")
use_custom = st.checkbox("🔧 Ввести значения вручную", value=False)

if not use_custom:
    fertilizer_type = st.selectbox("Выберите тип диаммофоски", options=[
        "10-26-26", "9-25-25", "12-24-12", "10-20-20"
    ])
    fertilizer_dict = {
        "10-26-26": (10, 26, 26),
        "9-25-25": (9, 25, 25),
        "12-24-12": (12, 24, 12),
        "10-20-20": (10, 20, 20)
    }
    n, p, k = fertilizer_dict[fertilizer_type]
    st.markdown(f"Азот (N): **{n}**, Фосфор (P₂O₅): **{p}**, Калий (K₂O): **{k}**")
else:
    n = st.slider("Азот (N)", 0, 100, 0, step=1)
    p = st.slider("Фосфор (P₂O₅)", 0, 100, 0, step=1)
    k = st.slider("Калий (K₂O)", 0, 100, 0, step=1)

# --------------------
# Вычисление штрафных признаков
# --------------------
acidic_soil = int(ph < 5.0)
alkaline_soil = int(ph > 8.0)
drought = int(rainfall < 150)
low_fertility = int((n + p + k) < 50)
low_moisture = int(moisture < 20)
min_temp_extreme = int(temp_min < -10.0)
max_temp_extreme = int(temp_max > 35.0)

# --------------------
# Отображение штрафных признаков
# --------------------
st.subheader("3. Штрафные признаки (автоматически)")
st.markdown(f"- Кислая почва (pH < 4.5): **{acidic_soil}**")
st.markdown(f"- Щелочная почва (pH > 8.0): **{alkaline_soil}**")
st.markdown(f"- Засуха (осадки < 150 мм): **{drought}**")
st.markdown(f"- Недостаток удобрений (N+P+K < 50): **{low_fertility}**")
st.markdown(f"- Недостаточная влажность (< 20%): **{low_moisture}**")
st.markdown(f"- Сильные заморозки (tmin < -10°C): **{min_temp_extreme}**")
st.markdown(f"- Экстремальный перегрев (tmax > 35°C): **{max_temp_extreme}**")

# --- Входной вектор ---
input_data = np.array([[
    field_id, temp_min, temp_avg, temp_max,
    rainfall, ph, moisture,
    n, p, k,
    acidic_soil, alkaline_soil,
    drought, low_fertility, low_moisture
]])

# --- Масштабирование ---
scaled_input = scaler.transform(input_data)

# Логирование в CSV
LOG_PATH = "outlier_log.csv"
log_entry = pd.DataFrame([[
field_id, temp_min, temp_avg, temp_max, rainfall, ph, moisture,
n, p, k
]], columns=[
"field_id", "temp_min", "temp_avg", "temp_max", "rainfall", "ph", "moisture",
"N", "P", "K"
])
log_entry.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

# --- Прогноз ---
if st.button("🔍 Рассчитать урожайность"):
    prediction = model.predict(scaled_input)[0][0]
    st.success(f"🌱 Урожайность: **{prediction:.2f} т/га**")
    st.info(f"⚠️ Учтено штрафов: **{int(np.sum(input_data[0][10:]))} из 5**")

    # --------------------
    # 🚨 Проверка на подозрительные значения
    # --------------------
    warnings = []

    if temp_max > 35:
        warnings.append("Максимальная температура слишком высокая (≥ 35°C)")
    if temp_min < -4:
        warnings.append("Минимальная температура слишком низкая (≤ -4°C)")
    if ph < 4.5 or ph > 8.0:
        warnings.append("Кислотность почвы вне диапазона (4.5–8.0)")
    if rainfall < 50 or rainfall > 500:
        warnings.append("Сумма осадков аномальная (менее 50 или более 500 мм)")
    if moisture < 5 or moisture > 50:
        warnings.append("Влажность почвы вне диапазона (5%–50%)")
    if (n + p + k) > 70:
        warnings.append("Суммарное количество удобрений слишком большое (> 70 кг/га)")

    # Показываем предупреждение на экране
    if warnings:
        st.warning(
            "⚠️ ВНИМАНИЕ: Обнаружены подозрительные значения входных параметров! "
            "Прогноз может быть недостоверным:\n\n" +
            "\n".join([f"- {w}" for w in warnings])
        )

    # -------------------
    # График: вклад по категориям
    # -------------------
    factors = ['Температура', 'Почва', 'Удобрения', 'Штрафы']

    # Определяем количество штрафных признаков
    num_penalties = int(np.sum(input_data[0][10:15]))  # если 5 штрафных!

    values = [
        temp_avg + (temp_max - temp_min) / 2 + rainfall / 50,
        ph + moisture / 5,
        (n + p + k) / 10,
        -num_penalties * 2
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(factors, values, color=['#4CAF50', '#FFC107', '#2196F3', '#F44336'])
    ax.set_ylabel("Условный вклад")
    ax.set_title("Визуализация вклада факторов")
    st.pyplot(fig)