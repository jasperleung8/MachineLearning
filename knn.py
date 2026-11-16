import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("dataSets/Iris.csv")
print(data.info())

X = data[["sepal_length","sepal_width","petal_length","petal_width"]]
y = data["species"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=0.3,random_state=43)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xTrain,yTrain)

pred = knn.predict(xTest)
confusionMatrix = confusion_matrix(yTest,pred)
classificationReport = classification_report(yTest,pred)
print(confusionMatrix)
print(classificationReport)

import numpy as np
from sklearn.metrics import f1_score

n = int(np.sqrt(xTrain.shape[0]))
f1scores = []

for i in range(1,n):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xTrain,yTrain)

    pred = knn.predict(xTest)
    f = f1_score(yTest,pred,average="macro")
    f1scores.append(f)
highestF1 = max(f1scores)
bestk = f1scores.index(highestF1)+1

print(bestk)


