from sklearn import datasets
#from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
#
# Load the Boston Data Set
#
bd = pd.read_csv('boston.csv')
X = bd.values.copy()
y = bd.keys().copy()
#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split( X[:, :-1], X[:, -1],train_size=0.7, random_state=42)
#
# Create an instance of Lasso Regression implementation
#
lr = linear_model.LinearRegression()
rr = linear_model.Ridge(alpha=0.01)
rr100 = linear_model.Ridge(alpha=100)
lasso = linear_model.Lasso(alpha=1.0)
lasso5 = linear_model.Lasso(alpha=5.0)
lasso20 = linear_model.Lasso(alpha=20.0)
#
# Fit the Lasso model
#
lr.fit(X_train, y_train)
rr.fit(X_train, y_train)
rr100.fit(X_train, y_train)
lasso.fit(X_train, y_train)
lasso5.fit(X_train, y_train)
lasso20.fit(X_train, y_train)
#
# Create the model score
#
lasso.score(X_test, y_test), lasso.score(X_train, y_train)
print(lr.coef_)
print(rr.coef_)
print(rr100.coef_)

print(lasso.coef_)
print(lasso5.coef_)
print(lasso20.coef_)