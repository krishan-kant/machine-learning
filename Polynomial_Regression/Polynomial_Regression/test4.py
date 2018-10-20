# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:23:56 2018
@author: Krishan
"""

# Polynomial Regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Splitting the data set into training set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""

# fitting linear regression into the dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# fitting polynomial regression into the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2  = LinearRegression()
lin_reg2.fit(X_poly,y)

"""# Visualsing the lineaer regression results
plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X))
plt.title("Truth or bluff")
plt.xlabel("Position level")
plt.ylabel("Salaries")
plt.show()
"""

# Visualising the polynomial regresson results
X_grid = np.arange(min(X),max(X),0.1)
X_grid =  X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y)
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)))
plt.title("Salary vs position")
plt.xlabel("Position level")
plt.ylabel("Salaries")
plt.show() 


# predicting the salary at level 6.5 with linear results
check_linear = lin_reg.predict(6.5)

# predicting salary with polynomial result
check_pol = lin_reg2.predict((poly_reg.fit_transform(6.5)))