import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("dataSets/car.csv")
print(data.info())
print(data.isnull().sum())
print(data["class"].value_counts())

for col in data.columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

print(data.info())

X = data[["sales","maintenance","doors","persons","boot_space","safety"]]
y = data["class"]

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)

tree = DecisionTreeClassifier()
tree.fit(xtrain,ytrain)

pred = tree.predict(xtest)
report = classification_report(ytest,pred)
matrix = confusion_matrix(ytest,pred)

print(matrix)
print(report)

