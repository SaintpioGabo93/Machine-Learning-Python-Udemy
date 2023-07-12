# Regresión Polinómica
# La refresión polinomica se usa cuando los datos tienen patrones exponenciales

# Importar Librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el conjunto de datos

conjuntoDatos = pd.read_csv('Position_Salaries.csv') # Este metodo hace una variable con el conjunto de datos
X = conjuntoDatos.iloc[:, 1:-1].values# Como ya está codificado el puesto, entonces no hace falta ponerlo por eso 1:-1, recordar que es un vector columna
y = conjuntoDatos.iloc[:,-1].values # Obtiene la variable dependiente, vector fila

print(X)
print('')
print(y)


# ------------------------  Entrenamiento del Modelo --------------------------- #

# Entrenar el modelo de regresión lineal para el conjunto de datos
# Importante, este es para comparar y ver la diferencia entre la regresión lineal y Polinómica

from sklearn.linear_model import LinearRegression

regresorLineal = LinearRegression()
regresorLineal.fit(X, y)

# Entrenamiento con el modelo de regresión lineal polinómica en el conjunto de datos

from sklearn.preprocessing import PolynomialFeatures

regresionPolinomica = PolynomialFeatures(degree = 8)
X_polinom = regresionPolinomica.fit_transform(X)
regresorLineal_2 = LinearRegression()
regresorLineal_2.fit(X_polinom, y)

# ---------------------- Fin del Entrenamiento ---------------------------- #

# Visualización de los resultados de la regresión lineal

plt.scatter(X, y, color = 'red')
plt.plot(X, regresorLineal.predict(X), color = 'blue')
plt.title('Mentira o verdad del Salario en Entrevista[Regresión lineal]')
plt.xlabel('Nivel de posición')
plt.ylabel('Salario')
plt.show()

# Visualización de los resultados con la regresión polinómica

plt.scatter(X, y, color = 'red')
plt.plot(X, regresorLineal_2.predict(regresionPolinomica.fit_transform(X)), color = 'blue')
plt.title('Mentira o verdad del Salario en Entrevista [Regresión Polinómica]')
plt.xlabel('Nivel de posición')
plt.ylabel('Salario')
plt.show()

# Visualización de la grafica con mejor resolución

X_red = np.arange(min(X), max(X), 0.1) # este determina la resolución
X_red = X_red.reshape((len(X_red), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_red, regresorLineal_2.predict(regresionPolinomica.fit_transform(X_red)), color = 'blue')
plt.title('Mentira o verdad del Salario en Entrevista [Regresión Polinómica], Mayor resolución')
plt.xlabel('Nivel de posición')
plt.ylabel('Salario')
plt.show()


# ------------------------------ Predicciones -------------------------- #
print('')
# Regresión Lineal
print(regresorLineal.predict([[6.5]])) # Es un vector bidimensional así
print('')
# Regresión Polinómica
print(regresorLineal_2.predict(regresionPolinomica.fit_transform([[6.5]])))