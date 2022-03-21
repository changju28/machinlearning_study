from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# 선형 회귀
"""
# sklearn 안에 있는 보스턴 데이터를 가져온다
boston_dataset = load_boston()

# print(boston_dataset.DESCR)

# print(boston_dataset.feature_names)
# print(boston_dataset.data.shape)

# print(boston_dataset.target)
# print(boston_dataset.target.shape)

# x 변수에 보스턴의 dataframe을 넣어준다
x = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# dataframe 의 age column을 x에 초기화 한다
x = x[['AGE']]
# print(x)

# y에 집값에 관련된 목표데이터를 넣어준다
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])
# print(y)

# x,y 의 훈련 데이터와 테스트 데이터를 나누어 넣어준다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 선형 회귀 알고리즘을 불러온다
model = LinearRegression()

# 모델에 훈련데이터를 넣어 훈련 시킨다.
model.fit(x_train, y_train)
# print(model.coef_)
# print(model.intercept_)

# 훈련된 모델의 가설함수에 나이 관련 테스트 데이터를 넣어 y값을 구한다.
y_test_prediction = model.predict(x_test)

# print(y_test_prediction)

# 구한 y값과 실제 y값을 비교한다.
print(mean_squared_error(y_test, y_test_prediction) ** 0.5)
"""

# 다중 선형 회귀
"""
# 보스턴 데이터 불러오기
boston_dataset = load_boston()

# print(boston_dataset.feature_names)

X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# print(X)

y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = LinearRegression()

model.fit(x_train, y_train)

y_test_prediction = model.predict(x_test)

print(mean_squared_error(y_test, y_test_prediction) ** 0.5)
"""

# 다항 회귀
"""
boston_dataset = load_boston()

# print(boston_dataset)
# print(boston_dataset.data.shape)

polynomial_transformer = PolynomialFeatures(2)

polynomial_data = polynomial_transformer.fit_transform(boston_dataset.data)

# print(polynomial_data.shape)

polynomial_feature_names = polynomial_transformer.get_feature_names(boston_dataset.feature_names)

# print(polynomial_feature_names)

X = pd.DataFrame(polynomial_data, columns=polynomial_feature_names)

# print(X)

y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])

# print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(x_test.shape)

model = LinearRegression()

model.fit(x_train, y_train)

# print(model.coef_)

# print(model.intercept_)
y_test_prediction = model.predict(x_test)
#
print(mean_squared_error(y_test, y_test_prediction) ** 0.5)
"""

# 로지스틱 회귀

iris_data = load_iris()

# print(iris_data.DESCR)

# X에 데이터를 넣는다
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# print(X)

# y에 결과값을 넣는다
y = pd.DataFrame(iris_data.target, columns=['class'])

# print(y)

# x ,y 의 훈련데이터와 결과데이터를 넣는다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

y_train = y_train.values.ravel()

# 로지스틱 회귀 모델을 생성
model = LogisticRegression(solver='saga', max_iter=2000)

# 훈련데이터로 모델을 훈련
model.fit(X_train, y_train)

# 훈련된 모델을 테이스 데이터를 넣어 결과 값을 얻는다
model.predict(X_test)

# 테스트 데이터를 넣은 모델을 채점 한다.
print(model.score(X_test, y_test))
