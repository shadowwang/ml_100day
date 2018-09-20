import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataSet = pd.read_csv('../datasets/50_Startups.csv')
X = dataSet.iloc[:,:-1].values
Y = dataSet.iloc[: , 4].values

labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[: , 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toArray()

X = X[: , 1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regression = LinearRegression()
regression.fit(X_train, Y_train)

y_pred = regression.predict(X_test)
