# Regresión lineal Simple

# Importar las librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Imporatr conjunto de datos

conjuntoDatos = pd.read_csv('Salary_Data.csv')
X = conjuntoDatos.iloc[:, :-1].values
y = conjuntoDatos.iloc[:, -1]

print(X)
print(y)
print('')

# Separar el conjunto de datos en el conjunto de entrenamiento y el conjunto de prueba

from sklearn.model_selection import train_test_split
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_entrenamiento)
print(X_prueba)
print(y_entrenamiento)
print(y_prueba)

# Entrenar la regresion lineal simple en el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(X_entrenamiento,y_entrenamiento)

# Predecir el conjunto de prueba

y_prediccion = regresor.predict(X_prueba)

# Visualizar el resultado del conjunto de entrenamiento

plt.scatter(X_entrenamiento, y_entrenamiento, color = 'red') # Esta grafica en puntos cada un de los datos del conjunto de entrenamiento
plt.plot(X_entrenamiento, regresor.predict(X_entrenamiento), color = 'blue') # Este muestra la linea resultante de la regresión
plt.title('Salario vs. Experiencia (Conjunto de entrenamiento)') #Pone título a la gráfica
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')

plt.show()

# Visualización de el conjunto de prueba

plt.scatter(X_prueba, y_prueba, color = 'red')
plt.plot(X_entrenamiento, regresor.predict(X_entrenamiento), color = 'blue')
plt.title('Salario vs. Experiencia (Conjunto de prueba)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')

plt.show()
