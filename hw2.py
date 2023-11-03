import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import datasets

def createPlot(x, y, xlabel, ylabel, title, filename, color):
    plt.scatter(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

# Create a dictionary to store the results
results = {
    "Model": [],
    "Lambda": [],
    "R^2": [],
    "MSE": [],
    "MAE": []
}

# Load the Boston housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

# Create a DataFrame to store the data
data = pd.DataFrame(data)
data.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO","B", "LSTAT"]
data['PRICE'] = raw_df.values[1::2, 2]

X = data.drop("PRICE", axis=1)
y = data["PRICE"]

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create Linear Regression model
lr = LinearRegression()

# Train the model using the training data
lr.fit(X_train, y_train)

# Model Evaluation
y_test_pred = lr.predict(X_test)

results["Model"].append("Linear Regression (Testing Set)")
results["Lambda"].append("N/A")
results["R^2"].append(r2_score(y_test, y_test_pred))
results["MSE"].append(mean_squared_error(y_test, y_test_pred))
results["MAE"].append(mean_absolute_error(y_test, y_test_pred))

createPlot(y_test, y_test_pred, "Actual Price", "Predicted Price", "Linear Regression (Testing Set)", "linreg/lr.png", 'blue')

# Ridge and Lasso Regression with different lambdas
alphas = [0.1, 1, 5, 50]

for alpha in alphas:

    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results["Model"].append("Ridge Regression")
    results["Lambda"].append(alpha)
    results["R^2"].append(r2_score(y_test, y_pred))
    results["MSE"].append(mean_squared_error(y_test, y_pred))
    results["MAE"].append(mean_absolute_error(y_test, y_pred))

    createPlot(y_test, y_pred, "Actual Price", "Predicted Price", f'Ridge Regression (Lambda={alpha})', f'ridge/ridge-{alpha}.png', 'green')

    # LASSO Regression
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    results["Model"].append("LASSO Regression")
    results["Lambda"].append(alpha)
    results["R^2"].append(r2_score(y_test, y_pred))
    results["MSE"].append(mean_squared_error(y_test, y_pred))
    results["MAE"].append(mean_absolute_error(y_test, y_pred))

    createPlot(y_test, y_pred, "Actual Price", "Predicted Price", f'LASSO Regression (Lambda={alpha})', f'lasso/lasso-{alpha}.png', 'red')

# Create a DataFrame to display the results

results_df = pd.DataFrame(results)
print(results_df)