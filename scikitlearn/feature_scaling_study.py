import pandas as pd
import numpy as np

from sklearn import preprocessing


# min-max-normelization
"""
nba_player_of_the_week_df = pd.read_csv('data/NBA_player_of_the_week.csv')

# print(nba_player_of_the_week_df.head())

# print(nba_player_of_the_week_df.describe())

height_weight_age_df = nba_player_of_the_week_df[['Height CM', 'Weight KG', 'Age']]

# print(height_weight_age_df)

scaler = preprocessing.MinMaxScaler()

normalized_data = scaler.fit_transform(height_weight_age_df)

print(normalized_data)

height_weight_age_df = pd.DataFrame(normalized_data, columns=[['Height CM', 'Weight KG', 'Age']])

print(height_weight_age_df.describe())
"""

# 36.2

#standardize (표준화)
"""

nba_player_of_the_week_df = pd.read_csv('data/NBA_player_of_the_week.csv')

scaler = preprocessing.StandardScaler()

height_weight_age_df = nba_player_of_the_week_df[['Height CM', 'Weight KG', 'Age']]

standardized_data = scaler.fit_transform(height_weight_age_df)

# pd.set_option('display.float_format', lambda x: '%.5' % x)

standardized_df = pd.DataFrame(standardized_data, columns=['Height', 'Weight', 'Age'])

print(standardized_df.describe())
"""

# min_max_normalzation 문제
# list = [25, 49, 32, 35, 40]
list = [25000000, 35000000, 30000000, 50000000, 35000000]
list1 = sorted(list)

min = list1[0]
max = list1[-1]


for i in list:
    print((i - min) / (max - min))

# ㄴ