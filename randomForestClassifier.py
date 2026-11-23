import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("dataSets/adult_income.csv")

print(data.info())
print(data.isnull().sum())

textColumns = data.select_dtypes(include=["object"]).columns
for col in textColumns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
print(data.info())

X = data.drop(columns="income")
y = data["income"]

