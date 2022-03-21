# 지니불순도
def gine(flu, not_flu, total):
    a = (flu / total) ** 2
    b = (not_flu / total) ** 2
    add = a + b
    return 1 - add
"""
yes = gine(15, 80, 95)
no = gine(90, 0, 90)


print(yes)
print(no)
total = (95 * yes + 90 * no) / 185

print(round(total, 3))
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris_data = load_iris()

# print(iris_data.DESCR)

X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# print(X)

y = pd.DataFrame(iris_data.target, columns=['class'])

# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = DecisionTreeClassifier(max_depth=4)

model.fit(X_train, y_train)

model.predict(X_test)

print(model.score(X_test, y_test))

importances = model.feature_importances_

indices_sorted = np.argsort(importances)

plt.figure()
plt.title('Feature importances')
plt.bar(range(len(importances)), importances[indices_sorted])
plt.xticks(range(len(importances)), X.columns[indices_sorted], rotation=90)
plt.show()