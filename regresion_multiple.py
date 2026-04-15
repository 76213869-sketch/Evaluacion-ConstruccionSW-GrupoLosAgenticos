import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# --- Carga de datos ---

# Asegúrate de que el archivo House_price.csv esté en el directorio correcto o ajusta la ruta
try:
    df_house = pd.read_csv('House_price.csv')
    print('Archivo House_price.csv cargado exitosamente.')
except FileNotFoundError:
    print('Error: El archivo House_price.csv no se encuentra en el directorio actual.')
    print('Asegúrate de que el archivo esté en /content/Evaluacion-ConstruccionSW-GrupoLosAgenticos/')
    df_house = pd.read_csv('/content/Evaluacion-ConstruccionSW-GrupoLosAgenticos/House_price.csv')
    print('Se cargó el archivo usando la ruta completa.')

# df_house.head() # Descomentar para ver las primeras filas (interactivo)
# df_house.info() # Descomentar para ver la información general (interactivo)
# df_house.describe() # Descomentar para ver estadísticas descriptivas (interactivo)

# --- Modelo de Regresión Lineal Múltiple (5 variables) ---

# Preparación de los datos para el Modelo 1 (5 variables)
numeric_cols = df_house.select_dtypes(include=np.number).columns.tolist()
if 'Price' in numeric_cols:
    numeric_cols.remove('Price')

X = df_house[numeric_cols]
y = df_house['Price']

# Manejar valores nulos (si los hubiera) imputando con la media
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].mean(), inplace=True)

# División de los datos en conjuntos de entrenamiento y prueba para el Modelo 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcción y entrenamiento del Modelo 1
model = LinearRegression()
model.fit(X_train, y_train)

print('')
print('--- Modelo de Regresión Lineal Múltiple (5 variables) ---')
print('Modelo entrenado.')
print(f'Coeficientes del modelo: {model.coef_}')
print(f'Intercepto del modelo: {model.intercept_}')

print('')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Visualización de Predicciones vs. Valores Reales (Modelo 1)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Precios Reales')
plt.ylabel('Precios Predichos')
plt.title('Precios Reales vs. Precios Predichos (Modelo 5 variables)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Línea de referencia y=x
plt.grid(True)
plt.show()

# Residuos del Modelo 1
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de los Residuos (Modelo 5 variables)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Precios Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Precios Predichos (Modelo 5 variables)')
plt.grid(True)
plt.show()

# --- Nuevo Modelo de Regresión Lineal Múltiple (3 variables) ---

# Seleccionar las 3 características principales
X_new = df_house[['Avg. Area Income', 'House Age', 'Number of Rooms']]
y_new = df_house['Price']

# Dividir los datos para el nuevo modelo
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Construir y entrenar el nuevo modelo
model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

print('')
print('--- Nuevo Modelo de Regresión Lineal Múltiple (3 variables) ---')
print('Modelo entrenado.')
print(f'Coeficientes del nuevo modelo: {model_new.coef_}')
print(f'Intercepto del nuevo modelo: {model_new.intercept_}')

print('')
print(f'Mean Squared Error (MSE): {mse_new:.2f}')
print(f'Mean Absolute Error (MAE): {mae_new:.2f}')
print(f'R-squared (R2): {r2_new:.2f}')

print('')
print('--- Comparación de Coeficientes ---')
print('Modelo Original (5 variables):')
for feature, coef in zip(X.columns, model.coef_):
    print(f'  {feature}: {coef:.2f}')

print('')
print('Nuevo Modelo (3 variables):')
for feature, coef in zip(X_new.columns, model_new.coef_):
    print(f'  {feature}: {coef:.2f}')

# Visualización 3D para el Modelo 2 (3 variables)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test_new['Avg. Area Income'], X_test_new['House Age'], y_test_new, color='blue', label='Precios Reales', alpha=0.6)
ax.scatter(X_test_new['Avg. Area Income'], X_test_new['House Age'], y_pred_new, color='red', label='Precios Predichos (Modelo 3 variables)', alpha=0.6)

ax.set_xlabel('Ingreso Promedio del Área')
ax.set_ylabel('Antigüedad de la Casa')
ax.set_zlabel('Precio')
ax.set_title('Precios Reales vs. Predichos (Modelo 3 variables) en 3D')
ax.legend()
plt.show()
