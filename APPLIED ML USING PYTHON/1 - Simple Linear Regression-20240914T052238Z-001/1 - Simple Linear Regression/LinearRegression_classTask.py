# Predicting Exam Scores of a student who studies for 6 hours

# import libraries

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('hours.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
plt.scatter(X, Y, color='red')
plt.title('Data Plot')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.show()

# Split the adataset into 80% training and 20% testing 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Training the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)

# Making Predictions
Y_pred = model.predict(X_test)

# Showing Accuracy of the model using
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate the model
print(f'Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred)}')
print(f'R² Score: {r2_score(Y_test, Y_pred)}')
