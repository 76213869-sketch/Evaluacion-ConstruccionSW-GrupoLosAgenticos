import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Regresión Polinómica Automática")

# Cargar datos
df = pd.read_csv('House_price.csv')

target_col = 'Price'

# Selección automática de variable
correlations = df.corr(numeric_only=True)[target_col].drop(target_col)
best_feature = correlations.abs().idxmax()

print(f"Variable seleccionada: {best_feature}")
print(f"Correlación: {correlations[best_feature]:.4f}\n")

X = df[[best_feature]].values
y = df[target_col].values

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Transformación polinómica
degree = 3
poly = PolynomialFeatures(degree)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Modelo
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predicciones
y_pred = model.predict(X_test_poly)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Métricas del Modelo ---")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}\n")

# Curva suave
X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
X_smooth_poly = poly.transform(X_smooth)
y_smooth = model.predict(X_smooth_poly)

# Gráfico
plt.figure(figsize=(12,7))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label="Datos reales")
plt.plot(X_smooth, y_smooth, color='red', linewidth=3, label="Modelo polinómico")

plt.xlabel(best_feature)
plt.ylabel(target_col)
plt.title(f"Regresión Polinómica: {target_col} vs {best_feature}")
plt.legend()
plt.grid(True)

plt.show()
