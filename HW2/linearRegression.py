import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model, feature_selection,preprocessing
from sklearn.model_selection import train_test_split
#import statsmodels.formula.api as sm
import statsmodels.api as sm
#from statsmodels.tools.eval_measures import mse
from statsmodels.tools.tools import add_constant
#from sklearn.metrics import mean_squared_error

sl_data = pd.read_csv('heightweight.csv')
X = sl_data.values.copy()
X_train, X_test, y_train, y_test = train_test_split( X[:, :-1], X[:, -1],train_size=0.80)
result = sm.OLS( y_train, add_constant(X_train) ).fit()

print(result.summary())

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
print("Coefficient of determination: %.2f" % r2_score(y_test, predicted))

rr = linear_model.Ridge(alpha=0.01)
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# in this case linear and ridge regression resembles
rr.fit(X_train, y_train)
rr100 = linear_model.Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)
train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

print("ridge regression alpha-0.01 train score: ", Ridge_train_score)
print("ridge regression alpha-0.01 test score: ", Ridge_test_score)
print("ridge regression alpha-100 train score: ", Ridge_train_score100)
print("ridge regression alpha-100 test score: ", Ridge_test_score100)

train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)

Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7)
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$')
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()
