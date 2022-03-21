import pandas as pd

# GENDER_FILE_PATH = 'datasets/gender.csv'

gender_df = pd.read_csv('datasets/ gender.csv')
input_data = gender_df.drop(['Gender'], axis=1)

# 여기 코드를 쓰세요
# print(gender_df)
# print(input_data)

X = pd.get_dummies(data=input_data)

pd.set_option('display.max.columns', None)

# 체점용 코드
print(X.head())