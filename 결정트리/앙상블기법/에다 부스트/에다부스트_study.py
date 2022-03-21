from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

iris_data = load_iris()

X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

y = pd.DataFrame(iris_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

y_train = y_train.values.ravel()

model = AdaBoostClassifier(n_estimators=100)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))
print(predictions)

print(model.feature_importances_)