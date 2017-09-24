# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:01:23 2017

@author: Harsh Mathur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                              
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

#predict result
y_pred = regressor.predict(6.5)

#visualise polynomial
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid ), color='blue')
plt.title('Random forest Regressor ')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()