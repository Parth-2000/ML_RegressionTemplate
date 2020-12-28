# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Training Model on the Dataset


# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
polynomial_regressor = LinearRegression()
polynomial_regressor.fit(X_poly, y_train)


# Multiple Linear Regression 
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor(random_state = 0)
decision_tree_regressor.fit(X_train, y_train)


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
random_forest_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
random_forest_regressor.fit(X_train, y_train)



# Predicting the test set result

# Polynomial Regression
y_pred_polynomial_regression = polynomial_regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_polynomial_regression.reshape(len(y_pred_polynomial_regression),1), y_test.reshape(len(y_test),1)),1))


# Multiple Linear Regression 
y_pred_linear_regression = linear_regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_linear_regression.reshape(len(y_pred_linear_regression),1), y_test.reshape(len(y_test),1)),1))


# Decision Tree Regression
y_pred_decision_tree_regression = decision_tree_regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_decision_tree_regression.reshape(len(y_pred_decision_tree_regression),1), y_test.reshape(len(y_test),1)),1))


# Random Forest Regression
y_pred_random_forest_regression = random_forest_regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_random_forest_regression.reshape(len(y_pred_random_forest_regression),1), y_test.reshape(len(y_test),1)),1))


# Evaluating the Model Performance

from sklearn.metrics import r2_score

print("<----------------------------------R^2 Score---------------------------------->")

print("Polynomial Regression (R^2 Score)      -----> ",r2_score(y_test, y_pred_polynomial_regression))

print("Multiple Linear Regression (R^2 Score) -----> ",r2_score(y_test, y_pred_linear_regression))

print("Decision Tree Regression (R^2 Score)   -----> ",r2_score(y_test, y_pred_decision_tree_regression))

print("Random Forest Regression (R^2 Score)   -----> ",r2_score(y_test, y_pred_random_forest_regression))


from sklearn.metrics import mean_squared_error
import math

print("<-------------------Mean Suared Error(MSE) & Root Mean Squared Error(RMSE)------------------->")

print("Polynomial Regression       -----> ", "MSE: ",mean_squared_error(y_test, y_pred_polynomial_regression), " SMSE: ",math.sqrt(mean_squared_error(y_test, y_pred_polynomial_regression)))

print("Multiple Linear Regression  -----> ", "MSE: ",mean_squared_error(y_test, y_pred_linear_regression), " SMSE: ",math.sqrt(mean_squared_error(y_test, y_pred_linear_regression)))

print("Decision Tree Regression    -----> ", "MSE: ",mean_squared_error(y_test, y_pred_decision_tree_regression), " SMSE: ",math.sqrt(mean_squared_error(y_test, y_pred_decision_tree_regression)))

print("Random Forest Regression    -----> ", "MSE: ",mean_squared_error(y_test, y_pred_random_forest_regression), " SMSE: ",math.sqrt(mean_squared_error(y_test, y_pred_random_forest_regression)))



from sklearn.metrics import mean_absolute_error


print("<-----------------------------Mean Absolute Error(MAE)---------------------------->")

print("Polynomial Regression (MAE)      -----> ",mean_absolute_error(y_test, y_pred_polynomial_regression))

print("Multiple Linear Regression (MAE) -----> ",mean_absolute_error(y_test, y_pred_linear_regression))

print("Decision Tree Regression (MAE)   -----> ",mean_absolute_error(y_test, y_pred_decision_tree_regression))

print("Random Forest Regression (MAE)   -----> ",mean_absolute_error(y_test, y_pred_random_forest_regression))





