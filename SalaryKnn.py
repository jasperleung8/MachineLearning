import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

data = pd.read_csv("dataSets/Salary.csv")
print(data.info())

X = data["YearsExperience"]
y = data["Salary"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X = np.array(X).reshape(-1,1)

xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=0.3,random_state=43)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(xTrain,yTrain)

pred = knn.predict(xTest)

rmse = root_mean_squared_error(yTest,pred)
print(rmse)
