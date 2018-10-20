# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:36:28 2018

@author: Admin
"""

# regression usiung decision tree

# importing libraries
import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset=pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# fitting decision tree regresion into the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# predicting the value for 6.5
y_pred = regressor.predict(6.5)

# Visualising the polynomial regresson results
X_grid = num.arange(min(X),max(X),0.1)
X_grid =  X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y)
plt.plot(X_grid,y)
plt.title("Position vs Salary")
plt.xlabel("Position level")
plt.ylabel("Salaries")
plt.show()