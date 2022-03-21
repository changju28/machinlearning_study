# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

diabetes_dataset = datasets.load_diabetes()  # 데이터 셋 갖고오기

# 코드를 쓰세요

polynomial_transformer = PolynomialFeatures(2)

polynomial_data = polynomial_transformer.fit_transform(diabetes_dataset.data)

polynomial_feature_name = polynomial_transformer.get_feature_names(diabetes_dataset.feature_names)

X = pd.DataFrame(polynomial_data, columns=polynomial_feature_name)

# 테스트 코드
print(X.head())