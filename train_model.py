import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

# --------------------
# 🔁 Фиксация случайности
# --------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --------------------
# 📥 Загрузка уже очищенных данных
# --------------------

df = pd.read_csv("processed_data_converted.csv")

print("✅ Загружен очищенный датасет")
print("Форма таблицы:", df.shape)
print("Дубликатов:", df.duplicated().sum())

# --------------------
# 🧾 Определение признаков и целевой переменной
# --------------------
X = df.drop(columns=["урожайность (т/га)"]).values
y = df["урожайность (т/га)"].values


# --------------------
# 🎟️ Добавим штрафные признаки
# --------------------
df['acidic_soil'] = (df['кислотность_повы (Ph)'] < 5.0).astype(int)
df['alkaline_soil'] = (df['кислотность_повы (Ph)'] > 8.0).astype(int)
df['drought'] = (df['rain_sum'] < 150).astype(int)
df['low_fertility'] = ((df['азот (N)'] + df['фосфор (P₂O₅)'] + df['калий (K₂O)']) < 50).astype(int)
df['low_moisture'] = (df['влажность_почвы'] < 20).astype(int)

# --------------------
# 📏 Масштабирование
# --------------------
X = df.drop(columns=['year', 'урожайность (т/га)'])
y = df['урожайность (т/га)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------
# 📊 Деление на выборки
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# --------------------
# 📉 Функция потерь (без штрафов)
# --------------------
loss_fn = tf.keras.losses.MeanSquaredError()

# --------------------
# 🧠 Архитектура модели с L2
# --------------------
def build_model(l2_value):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_value),
            input_shape=(X_train.shape[1],)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_value)
        ),
         tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])


# --------------------
# 🔁 Перебор по значениям регуляризации с ранней остановкой
# --------------------
best_mae = float('inf')
best_l2 = None
best_model = None
history = None

for l2_val in [0.00001]:
    print(f"\n🚀 Пробую L2-регуляризацию: {l2_val}")
    model = build_model(l2_val)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, restore_best_weights=True
    )

    hist = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=4,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"L2 = {l2_val}, MAE = {mae:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_l2 = l2_val
        best_model = model
        history = hist

print(f"\n✅ Лучшее значение L2-регуляризации: {best_l2}, MAE: {best_mae:.4f}")
model = best_model

# --------------------
# 📈 Оценка
# --------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"\nСредняя абсолютная ошибка (MAE): {mae:.2f} т/га")
print(f"Среднеквадратичная ошибка: {loss:.2f}")

# --------------------
# 🤖 Предсказания
# --------------------
y_pred = model.predict(X_test).flatten()
y_test = y_test.reset_index(drop=True)
print("y_test[:10]:", y_test[:10])
print("y_pred[:10]:", y_pred[:10])
print("min/max предсказаний:", np.min(y_pred), "/", np.max(y_pred))

# --------------------
# 🧮 Метрики
# --------------------
r2 = r2_score(y_test, y_pred)
print(f"Коэффициент детерминации R²: {r2:.3f}")
errors = y_test - y_pred
for i in range(len(y_test)):
    print(f"Факт: {y_test[i]:.2f} → Прогноз: {y_pred[i]:.2f} → Ошибка: {errors[i]:.2f}")


# --------------------
# 📊 График распределения ошибок
# --------------------
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20, color='purple', edgecolor='black')
plt.title("Распределение ошибок прогноза")
plt.xlabel("Ошибка (факт - прогноз)")
plt.ylabel("Количество случаев")
plt.grid(True)
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.show()

# --------------------
# 📊 График: факт vs прогноз
# --------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Фактическая урожайность (т/га)")
plt.ylabel("Прогнозируемая урожайность (т/га)")
plt.title("Сравнение факта и прогноза")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_vs_actual.png")
plt.show()

# --------------------
# 📉 График потерь
# --------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Обучающая')
plt.plot(history.history['val_loss'], label='Валидация')
plt.xlabel("Эпоха")
plt.ylabel("Потери (MSE)")
plt.title("График потерь по эпохам")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# --------------------
# 💾 Сохранение
# --------------------
model.save("yield_model.h5")
joblib.dump(scaler, "scaler.save")

print("✅ Модель успешно обучена и сохранена.")

# --------------------
# 📄 Сохранение отчёта
# --------------------
with open("training_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Отчёт о модели ===\n")
    f.write(f"Средняя абсолютная ошибка (MAE): {mae:.2f} т/га\n")
    f.write(f"Среднеквадратичная ошибка (MSE): {loss:.2f}\n")
    f.write(f"Коэффициент детерминации R²: {r2:.3f}\n")
    f.write(f"Количество эпох обучения: {len(history.history['loss'])}\n")
    f.write("✅ Модель обучена и сохранена.\n")