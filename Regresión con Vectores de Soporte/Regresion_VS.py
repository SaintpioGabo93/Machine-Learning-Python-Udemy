# Regresión con Vectores de Soporte

# Importar Librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el conjunto de datos

conjuntoDatos = pd.read_csv('Position_Salaries.csv')
X = conjuntoDatos.iloc[:, 1:-1]
y = conjuntoDatos.iloc[:,-1]
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

# Escalarod de Características "Feature Scaling"

from sklearn.preprocessing import StandardScaler

escaladorEstandar_X = StandardScaler()
escaladorEstandar_y= StandardScaler()
X = escaladorEstandar_X.fit_transform(X)
y = escaladorEstandar_y.fit_transform(y)

print(X)
print(y)

# Entrenamiento del modelo Regresión con Vectores de Soporte en el Conjunto de Datos

from sklearn.svm import SVR

regresor = SVR(kernel = 'rbf')
regresor.fit(X, y)

# Predecir un nuevo resultado

escaladorEstandar_y.inverse_transform(regresor.predict(escaladorEstandar_X.transform([[6.5]])).reshape(-1,1))

# Visualizar los resultados de la SVR

plt.scatter(escaladorEstandar_X.inverse_transform(X),escaladorEstandar_y.inverse_transform(y), color = 'red')
plt.plot(escaladorEstandar_X.inverse_transform(X), escaladorEstandar_y.inverse_transform(regresor.predict(X).reshape(-1,1)),color = 'blue')
plt.title('Verdad o Mentira Salario')
plt.xlabel('Nivel de Posición')
plt.ylabel('Salario')
plt.show()

# Visualizando los resultados con una curva de mejor Resolución

X_red = np.arange(min(escaladorEstandar_X.inverse_transform(X)), max(escaladorEstandar_X.inverse_transform(X)), 0.1) # este determina la resolución
X_red = X_red.reshape((len(X_red), 1))
plt.scatter(escaladorEstandar_X.inverse_transform(X), escaladorEstandar_y.inverse_transform(y), color = 'red')
plt.plot(X_red, escaladorEstandar_y.inverse_transform(regresor.predict(escaladorEstandar_X.transform(X_red)).reshape(-1,1)) , color = 'blue')
plt.title('Mentira o verdad del Salario en Entrevista [Regresión Polinómica], Mayor resolución')
plt.xlabel('Nivel de posición')
plt.ylabel('Salario')
plt.show()


