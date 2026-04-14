import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('House_price.csv')

target_col = 'Price'

correlations = df.corr(numeric_only=True)[target_col].drop(target_col)
best_feature = correlations.abs().idxmax()

print(f"Variable independiente seleccionada automáticamente: '{best_feature}'")
print(f"Correlación con {target_col}: {correlations[best_feature]:.4f}\n")

X = df[[best_feature]]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Métricas del Modelo ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente R²: {r2:.4f}\n")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label='Datos reales (Test)')
plt.plot(X_test, y_pred, linewidth=2, label='Línea de tendencia (Modelo)')
plt.xlabel(best_feature)
plt.ylabel(target_col)
plt.title(f'Regresión Lineal Simple: {target_col} vs {best_feature}')
plt.legend()
plt.grid(True)
plt.show()
