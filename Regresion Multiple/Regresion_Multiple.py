 # Regresion lineal Múltiple

# Importación de las librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el conjunto de datos

conjuntoDatos = pd.read_csv('50_Startups.csv')
X = conjuntoDatos.iloc[:,:-1].values # Este método extrae los valores para la matriz de variables indendientes
y = conjuntoDatos.iloc[:,-1].values # Este método extrae los valores para la matriz de variable dependiente

print(X)
print('')
print(y)
print('')

# Codificación de las categorías Como es una regresión Multiple será un OneHot Encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformadorColumna = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [3])],remainder = 'passthrough' ) # Este metodo transforma las variables alfanuméricas en etiquetas vectoriales
X = np.array(transformadorColumna.fit_transform(X)) # Pasa por toda la matriz para convertir las entradas a etiquetas vectoriales

# Separar el conjunto de datos en un conjunto de entrenamiento y conjunto de prueba

from sklearn.model_selection import train_test_split

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X ,y , test_size= 0.2, random_state=0)

print(X_entrenamiento)
print('')
print(X_prueba)
print('')
print(y_entrenamiento)
print('')
print(X_prueba)
print('')

# Entrenamiento del modelo

from sklearn.linear_model import LinearRegression

regresor = LinearRegression()
regresor.fit(X_entrenamiento,y_entrenamiento)

# Predecier los resultados de la prueba

y_predecida = regresor.predict(X_prueba)
np.set_printoptions(precision=2)
print(np.concatenate((y_predecida.reshape(len(y_predecida),1), y_prueba.reshape(len(y_prueba),1)),1))

