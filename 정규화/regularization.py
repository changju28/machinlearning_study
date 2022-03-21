from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from math import sqrt

import pandas as pd
import numpy as np
admission_df = pd.read_csv('dataset/admission_data.csv').drop('Serial No.', axis=1)

# print(admission_df.head())

# 과적합 실행 예시
"""
X = admission_df.drop(['Chance of Admit '], axis=1)

polynomial_transformer = PolynomialFeatures(6)

polynomial_features = polynomial_transformer.fit_transform(X.values)

features = polynomial_transformer.get_feature_names(X.columns)

X = pd.DataFrame(polynomial_features, columns=features)

print(X.head())

y = admission_df[['Chance of Admit ']]

print(y.head())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = LinearRegression()
model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

mse = mean_squared_error(y_train, y_train_predict)

print('training set 에서의 성능')
print('=======================')
print(sqrt(mse))

mse = mean_squared_error(y_test, y_test_predict)

print('test set 에서의 성능')
print('=======================')
print(sqrt(mse))
"""

# 정규화

X = admission_df.drop(['Chance of Admit '], axis=1)

polynomial_transformer = PolynomialFeatures(6)

polynomial_features = polynomial_transformer.fit_transform(X.values)

features = polynomial_transformer.get_feature_names(X.columns)

X = pd.DataFrame(polynomial_features, columns=features)

print(X.head())

y = admission_df[['Chance of Admit ']]

print(y.head())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# L1
model = Lasso(alpha=0.001, max_iter=1000, normalize=True)
# L2
# model = Ridge(alpha=0.001, max_iter=1000, normalize=True)
model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

mse = mean_squared_error(y_train, y_train_predict)

print('training set 에서의 성능')
print('=======================')
print(mse ** 0.5)

mse = mean_squared_error(y_test, y_test_predict)

print('test set 에서의 성능')
print('=======================')
print(sqrt(mse))

