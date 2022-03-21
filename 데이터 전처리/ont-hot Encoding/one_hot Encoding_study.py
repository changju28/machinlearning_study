import pandas as pd

titanic_df = pd.read_csv('datasets/titanic.csv')

pd.set_option('display.max_columns', None)

# print(titanic_df.head())

titanic_sex_embarked = titanic_df[['Sex', 'Embarked']]

# print(titanic_sex_embarked)

# one_hot_encoding_df = pd.get_dummies(titanic_sex_embarked)

# print(one_hot_encoding_df.head())

one_hot_encoding_df = pd.get_dummies(data=titanic_df, columns=['Sex', 'Embarked'])


print(one_hot_encoding_df)