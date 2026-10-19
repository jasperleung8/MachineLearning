import matplotlib.pyplot as plt
import numpy as np


x = np.arange(1,11)
y = np.array([10,18,12,11,10,12,11,15,17,13])

plt.scatter(x,y)
plt.show()

#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)

meanX = np.mean(x)
meanY = np.mean(y)
m = np.sum((x-meanX)*(y-meanY)) / np.sum((x-meanX)**2)
c = meanY - m * meanX

print("Slope",m)
print("intercp",c)

#Prediction

p = m*x+c

print(p)

plt.scatter(x,y)
plt.plot(x,p,"r-o")
plt.show()

#estimating error
#RMSE - Root Mean Squared Error sqrt( sum( (p – yi)^2 )/n )

rmse = np.sqrt(np.mean((p-y)**2))

print("error",rmse)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

newX = x.reshape(-1,1)
lr.fit(newX,y)

print("m = ",lr.coef_)
print("c =",lr.intercept_)

predy = lr.predict(newX)
print(predy)

from sklearn.metrics import root_mean_squared_error

error = root_mean_squared_error(y,predy)
print(error)