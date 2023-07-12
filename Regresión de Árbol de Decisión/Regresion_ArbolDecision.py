# Regresión de Arbol de Decisión


# Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar Conjunto de Datos

conjuntoDatos = pd.read_csv('Position_Salaries.csv')
X = conjuntoDatos.iloc[:, 1:-1].values
y = conjuntoDatos.iloc[:, -1].values

# Entrenamietno del modelo de Regresión de Arbol de Decisión

from sklearn.tree import DecisionTreeRegressor
regresor = DecisionTreeRegressor(random_state = 0)
regresor.fit(X, y)

# Predicción del nuevo resultado

pred = regresor.predict([[6.5]])
print(pred)


# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regresor.predict(X_grid), color = 'blue')
plt.title('Verdad o Mentira(Decision Tree Regression)')
plt.xlabel('Nivel de Posición')
plt.ylabel('Salario')
plt.show()