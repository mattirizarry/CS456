import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the Boston housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Initialize lists to store results
results = []

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
results.append(('Linear Regression', lr_r2, lr_mse, lr_mae))

# Ridge Regression with different lambda values
ridge_lambdas = [0.1, 1, 5, 50]
for lambda_val in ridge_lambdas:
    ridge_model = Ridge(alpha=lambda_val)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    results.append(('Ridge (lambda={})'.format(lambda_val), ridge_r2, ridge_mse, ridge_mae))

# LASSO Regression with different lambda values
lasso_lambdas = [0.1, 1, 5, 50]
for lambda_val in lasso_lambdas:
    lasso_model = Lasso(alpha=lambda_val)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    lasso_r2 = r2_score(y_test, lasso_pred)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    lasso_mae = mean_absolute_error(y_test, lasso_pred)
    results.append(('LASSO (lambda={})'.format(lambda_val), lasso_r2, lasso_mse, lasso_mae))

# Create a DataFrame to display the results
results_df = pd.DataFrame(results, columns=['Model', 'R^2', 'MSE', 'MAE'])

# Print the results
print(results_df)
