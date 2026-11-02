import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

x = np.array(range(1,21))
y = x**2 + 5

plt.plot(x,y)
plt.show()

data = pd.read_csv("dataSets/HousingData.csv")
print(data.info())
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.isnull().sum())

X = data[["RM","LSTAT"]]
y = data["MEDV"]

xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
xTrainPoly = poly.fit_transform(xTrain)
xTestPoly = poly.fit_transform(xTest)

print(xTrain[0:3])
print(xTrainPoly[0:3])

lr = LinearRegression()
lr.fit(xTrainPoly,yTrain)
print(lr.coef_)
print(lr.intercept_)

yPred = lr.predict(xTestPoly)
error = root_mean_squared_error(yTest,yPred)
print(error)

