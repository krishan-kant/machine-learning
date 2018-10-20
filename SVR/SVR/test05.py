# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:04:37 2018

@author: Admin
"""

# Using Regression in SVR

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2.values
y = dataset.iloc[:,2].values

# Splitting the data set into training set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))  
#X = X.reshape(-1,1)
#y = y.reshape(-1,1)
# fitting svm into the dataset

from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,y)
y_pred=regressor.predict(X[0,0])
print(y_pred)

"""# Visualsing the lineaer regression results
plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X))
plt.title("Truth or bluff")
plt.xlabel("Position level")
plt.ylabel("Salaries")
plt.show()
"""
"""
# Visualising the polynomial regresson results
X_grid = np.arange(min(X),max(X),0.1)
X_grid =  X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y)
plt.plot(X_grid,)
plt.title("Position vs Salary")
plt.xlabel("Position level")
plt.ylabel("Salaries")
plt.show() """