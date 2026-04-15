import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("🔥 Regresión Polinómica - Gráfico Profesional")

# Cargar datos
df = pd.read_csv("/content/House_price.csv")

# Variables
X = df[['Avg. Area Income']].values
y = df['Price'].values

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Polinómica
degree = 3
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Modelo
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predicción
y_pred = model.predict(X_test_poly)

# Métricas
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# 🔥 CURVA SUAVE
X_smooth = np.linspace(X.min(), X.max(), 400).reshape(-1,1)
X_smooth_poly = poly.transform(X_smooth)
y_smooth = model.predict(X_smooth_poly)

# 🎨 GRÁFICO PRO
plt.figure(figsize=(12,7))

# Datos entrenamiento
plt.scatter(X_train, y_train, 
            color='blue', alpha=0.4, label="Train")

# Datos prueba
plt.scatter(X_test, y_test, 
            color='orange', alpha=0.6, label="Test")

# Curva polinómica
plt.plot(X_smooth, y_smooth, 
         color='red', linewidth=3, label="Modelo Polinómico")

# Estética
plt.title("Regresión Polinómica (Curva Suave)", fontsize=16)
plt.xlabel("Avg. Area Income", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Fondo blanco limpio
plt.style.use('seaborn-v0_8')

plt.show()
