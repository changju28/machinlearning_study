from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

iris_data = load_iris()

X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()

model = RandomForestClassifier(n_estimators=100, max_depth=4)

model.fit(X_train, y_train)

predict = model.predict(X_test)

print(predict)
print(model.score(X_test, y_test))
print(model.feature_importances_)