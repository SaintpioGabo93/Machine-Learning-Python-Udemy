# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
conjuntoDatos = pd.read_csv('Position_Salaries.csv')
X = conjuntoDatos.iloc[:, 1:-1].values
y = conjuntoDatos.iloc[:, -1].values


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor

regresor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regresor.fit(X, y)

# Predicting a new result
pred = regresor.predict([[6.5]])
print(pred)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regresor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()