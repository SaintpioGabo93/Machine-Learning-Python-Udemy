# Preprocesamiento de Datos Pasos y procedimientos.

# Librerias utilizadas:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el conjunto de datos:

conjuntoDatos = pd.read_csv('Data.csv')
X = conjuntoDatos.iloc[:,:-1].values # Este método recaba todas la variables independientes de nuestro conjunto de datos
y = conjuntoDatos.iloc[:, -1].values # Este método recaba todos los datos que vamos a predecir

# Hacernos cargo de los datos faltantes:
from sklearn.impute import SimpleImputer

asignacion = SimpleImputer(missing_values = np.nan, strategy = 'mean') # Esta linea nos dice que para los valores que se encuentran en nan usará la estrategia de promedio para rellenar esos espacios
asignacion.fit(X[:, 1:3]) # Con esta linea estamos recorriendo el arreglo de datos que sean numéricos asignando el valor según la estrategia señalada en la linea anterior.
X[:, 1:3] = asignacion.transform(X[:, 1:3]) # Así ya que se recorrió el arreglo y se aplicó la estrategia indicada, procede a ingresar los datos obtenidos y asignarlos a las celdas vacías.

# Asígnar categoría a los datos (OneHot Encoding):

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

    # Para el Tensor X (Variable independiente)
trasformadorColumnas = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[0])], remainder = 'passthrough') # Esta linea de codigo convierte los valores tipo a etiquetas tipo 1,2,3, etc.
X = np.array(trasformadorColumnas.fit_transform(X)) # Pasa por el arreglo y convierte los valores a categorias en la columna 0

    # Para el Tensor y (Variable dependiente)

from sklearn.preprocessing import LabelEncoder

codificadorEtiquetas = LabelEncoder()
y = codificadorEtiquetas.fit_transform(y)

# Separar el conjunto de datos para el conjunto de entrenamiento y conjunto de prueba

from sklearn.model_selection import train_test_split

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size = 0.2, random_state=1) # Del curso de ML Arduino, sólo recordar este paso

# Escalamiento de Características

from sklearn.preprocessing import  StandardScaler

escaladorEstandar = StandardScaler()
X_entrenamiento[:, 3:] = escaladorEstandar.fit_transform(X_entrenamiento[:,3:]) # Esta linea dice que va a recorrer todas las filas, y luego va a recorrer todas las columnas a partir de la 3 [3:] para realizar el escalamiento Estandar de los datos numéricos
X_prueba[:,3:] =escaladorEstandar.transform(X_prueba[:,3:]) # En esta línea ya los asigna al Tensor
