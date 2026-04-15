
# ========================================
# PARTE 1: CLASIFICACIÓN CON IRIS (REAL)
# ========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. Cargar datos
df = pd.read_csv("Iris.csv")

# 2. Preparar datos
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Convertir etiquetas
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 7. Predicción
y_pred = modelo.predict(X_test)

# 8. Métricas
print("===== RESULTADOS IRIS =====")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))


# ========================================
# PARTE 2: VISUALIZACIÓN (SIMULACIÓN)
# ========================================

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Datos simulados
mean_0 = [3, 4]
cov_0 = [[0.2, 0], [0, 0.2]]
X_0 = np.random.multivariate_normal(mean_0, cov_0, 100)

mean_1 = [7, 6]
cov_1 = [[0.2, 0], [0, 0.2]]
X_1 = np.random.multivariate_normal(mean_1, cov_1, 100)

# Frontera
m = -2
c = 15

x_boundary = np.array([0, 10])
y_boundary = m * x_boundary + c

# Gráfico
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X_0[:, 0], X_0[:, 1], color='blue', marker='o', s=60, label='Clase 0')
ax.scatter(X_1[:, 0], X_1[:, 1], color='purple', marker='s', s=60, label='Clase 1')

ax.plot(x_boundary, y_boundary, color='black', linewidth=2.5, label='Decision Boundary')

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_ylim(2, 8)

plt.title("Visualización de Clasificación")
plt.legend()
plt.show()
