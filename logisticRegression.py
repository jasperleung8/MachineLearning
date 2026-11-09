import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv("dataSets/titanic(1).csv")
print(data.info())
print(data.isnull().sum())

X = data.drop(columns=["Name","Survived"])
y = data["Survived"]

encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])

trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.3,random_state=42)

lr = LogisticRegression()
lr.fit(trainX,trainy)

pred = lr.predict(testX)
print(pred)

confusionMatrix = confusion_matrix(testy,pred)
print(confusionMatrix)

sb.heatmap(confusionMatrix,annot=True,fmt="d")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

report = classification_report(testy,pred)
print(report)
