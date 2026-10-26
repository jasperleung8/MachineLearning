import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = pd.read_csv("dataSets/HousingData.csv")
print(data.info())
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.isnull().sum())

X = data[["RM","LSTAT"]]
y = data["MEDV"]

xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=0.3,random_state=42)

lr = LinearRegression()

lr.fit(xTrain,yTrain)
print("slope",lr.coef_)
print("intercp",lr.intercept_)

pTest = lr.predict(xTest)
pTrain = lr.predict(xTrain)

testError = root_mean_squared_error(yTest,pTest)
trainError = root_mean_squared_error(yTrain,pTrain)
print("min Y",y.min())
print("max Y",y.max())

print("test error",testError)
print("train error",trainError)



