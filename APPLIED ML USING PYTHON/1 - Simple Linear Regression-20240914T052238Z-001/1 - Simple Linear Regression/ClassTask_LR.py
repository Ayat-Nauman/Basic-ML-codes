# Class Task

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Students.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
plt.scatter(X, y, color = 'red')
plt.title('Data Plot')
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.show()

R_square = regressor.score(X_train, y_train)