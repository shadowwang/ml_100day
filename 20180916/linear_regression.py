import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as  mp

##1.数据预处理
dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[:, :1].values##时间
Y = dataset.iloc[:, 1].values##分数
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1/4, random_state=0)

##2.训练集使用简单线性回归处理

simple_lr = LinearRegression()
simple_lr = simple_lr.fit(X_train, Y_train)##训练数据

##3.预测结果
Y_pred = simple_lr.predict(X_test)

##4.可视化展示
mp.scatter(X_train, Y_train, color='red')
mp.plot(X_train, simple_lr.predict(X_train), color='blue')
mp.show()

mp.scatter(X_test, Y_test, color='red')
mp.plot(X_test, simple_lr.predict(X_test), color='blue')
mp.show()