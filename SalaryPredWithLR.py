import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataSets/Salary.csv")
data.info()

X = data["YearsExperience"]
y = data["Salary"]

lr = LinearRegression()

newX = np.array(X).reshape(-1,1)
lr.fit(newX,y)

print("slope",lr.coef_)
print("intercp",lr.intercept_)
