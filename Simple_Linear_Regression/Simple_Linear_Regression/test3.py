# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:06:27 2018

@author: Krishan
"""
# simple linear regression

# importing libraries
import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset=pd.read_csv('Salary_data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Splitting the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)

# fiitng the simple regression model into the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# prediucting the test set results
y_pred = regressor.predict(X_test)

# Visalizing the training set results
plt.scatter(X_train,y_train, color = "blue")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Salary vs Experience Training")
plt.xlabel("Years of experience")
plt.ylabel("Salaries")
plt.show()

# Visalizing the test set results
"""plt.scatter(X_test,y_test, color = "blue")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Salary vs Experience Test")
plt.xlabel("Years of experience")
plt.ylabel("Salaries")
plt.show()"""