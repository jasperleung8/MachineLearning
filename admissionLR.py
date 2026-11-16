import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv("dataSets/admission.csv")
print(data.info())
print(data.isnull().sum())

X = data[["gre","gpa","rank"]]
y = data["admit"]

print(y.value_counts())

underSampler = RandomUnderSampler(sampling_strategy="majority",random_state=42)
X,y = underSampler.fit_resample(X,y)

print(y.value_counts())

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
