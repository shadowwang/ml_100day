import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## 导入数据集
df = pd.read_csv('../datasets/Data.csv')
X = df.iloc[ : , :-1].values
Y = df.iloc[ : , 3].values
print("----------------")
print("step2:导入数据集")
print("X")
print(X)
print("Y")
print(Y)


imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])##取所有纬度数据里第1列到第3列的值
print("---------------------")
print("Step 3: 处理丢失数据")
print("X")
print(X)

labeEncoder_X = LabelEncoder()
X[:, 0] = labeEncoder_X.fit_transform(X[: , 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
labeEncoder_Y = LabelEncoder()
Y = labeEncoder_Y.fit_transform(Y)
print("----------------------")
print("Step 4: 解析分类数据")
print("X")
print(X)
print("Y")
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("----------------------")
print("Step 5: 拆分数据集为训练集合和测试集合")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print("----------------------")
print("Step 6: 特征量化")
print("X_train")
print(X_train)
print("X_test")
print(X_test)


