import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Cargar los datos
df = pd.read_csv('House_price.csv')

# Definir la columna objetivo
target_col = 'Price'

# Para un gráfico 3D, DEBEMOS usar exactamente 2 variables independientes
# (X, Y serán estas variables, y Z será el Precio)
independent_cols_3d = ['Avg. Area Income', 'House Age']

print(f"Variables seleccionadas para el modelo 3D: {independent_cols_3d}")

X_3d = df[independent_cols_3d]
y = df[target_col]

# Dividir los datos (usamos menos datos para que el gráfico 3D no se sature de puntos)
X_train, X_test, y_train, y_test = train_test_split(X_3d, y, test_size=0.1, random_state=42)

# 2. Entrenar el modelo con 2 variables
model_3d = LinearRegression()
model_3d.fit(X_train, y_train)

# 3. Preparar la visualización 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extraer las variables para los ejes
x1_test = X_test['Avg. Area Income']
x2_test = X_test['House Age']

# A. Graficar los datos reales como puntos (Scatter plot)
ax.scatter(x1_test, x2_test, y_test, color='teal', alpha=0.6, edgecolors='k', label='Datos reales')

# B. Crear el plano de predicción (Superficie 3D)
# Creamos una malla (grid) con los valores mínimos y máximos de nuestras 2 variables
x1_rango = np.linspace(X_3d['Avg. Area Income'].min(), X_3d['Avg. Area Income'].max(), 10)
x2_rango = np.linspace(X_3d['House Age'].min(), X_3d['House Age'].max(), 10)

# Convertimos los rangos en coordenadas 2D para el plano
x1_malla, x2_malla = np.meshgrid(x1_rango, x2_rango)

# Aplanamos la malla para que el modelo pueda hacer las predicciones
malla_plana = pd.DataFrame({
    'Avg. Area Income': x1_malla.ravel(),
    'House Age': x2_malla.ravel()
})

# Calculamos el Precio (Z) para cada punto de la malla
z_malla = model_3d.predict(malla_plana)
# Reconstruimos la forma de la malla para graficarla
z_malla = z_malla.reshape(x1_malla.shape)

# Graficamos el plano de regresión (las predicciones del modelo)
ax.plot_surface(x1_malla, x2_malla, z_malla, alpha=0.4, color='orange', edgecolor='none')

# 4. Etiquetas y configuración visual
ax.set_xlabel('Avg. Area Income')
ax.set_ylabel('House Age')
ax.set_zlabel('Price')
ax.set_title('Regresión Lineal Múltiple en 3D (2 Variables vs Precio)')

# Ajustar el ángulo de visión (elevación, azimut)
ax.view_init(elev=20, azim=-45)

plt.tight_layout()
plt.show()