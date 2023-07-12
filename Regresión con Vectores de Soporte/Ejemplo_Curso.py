# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
conjuntoDatos = pd.read_csv('Position_Salaries.csv')
X = conjuntoDatos.iloc[:, 1:-1].values
y = conjuntoDatos.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
escaladorEstandar_X = StandardScaler()
escaladorEstandar_y = StandardScaler()
X = escaladorEstandar_X.fit_transform(X)
y = escaladorEstandar_y.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regresor = SVR(kernel = 'rbf')
regresor.fit(X, y)

# Predicting a new result
escaladorEstandar_y.inverse_transform(regresor.predict(escaladorEstandar_X.transform([[6.5]])).reshape(-1,1))

# Visualising the SVR results
plt.scatter(escaladorEstandar_X.inverse_transform(X), escaladorEstandar_y.inverse_transform(y), color = 'red')
plt.plot(escaladorEstandar_X.inverse_transform(X), escaladorEstandar_y.inverse_transform(regresor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(escaladorEstandar_X.inverse_transform(X)), max(escaladorEstandar_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(escaladorEstandar_X.inverse_transform(X), escaladorEstandar_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, escaladorEstandar_y.inverse_transform(regresor.predict(escaladorEstandar_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()