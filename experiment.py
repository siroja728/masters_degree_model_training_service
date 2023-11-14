import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Генерація даних для апроксимації
np.random.seed(0)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel()

# Створення моделі
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=1, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1, activation="linear"))

# Компіляція моделі
model.compile(loss="mean_absolute_error", optimizer="adam")

# Навчання моделі
model.fit(X, y, epochs=1000, batch_size=20, verbose=0)

# Передбачення для виведення графіку
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = model.predict(X_test)

# Графік результатів
plt.scatter(X, y, color="black", label="Дані")
plt.plot(X_test, y_pred, color="blue", linewidth=3, label="Апроксимація")
plt.title("Багатошаровий персептрон для апроксимації функції")
plt.legend()
plt.show()
