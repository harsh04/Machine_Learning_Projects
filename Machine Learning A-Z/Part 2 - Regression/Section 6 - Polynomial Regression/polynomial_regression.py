# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:59:09 2017

@author: Harsh Mathur
"""

#polynomial regressors
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fitting polynomial to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualise linear
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualise polynomial
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid )), color='blue')
plt.title('Polynomial')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predict linear
lin_reg.predict(6.5)

#predict poly
lin_reg_2.predict(poly_reg.fit_transform(6.5))
