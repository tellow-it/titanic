import pandas as pd

print('-----------------------')
data = pd.read_csv("train.csv")

print(data.isnull().sum())
data['Age'] = data['Age'].fillna((data['Age'].median()))
data['Cabin'] = data['Cabin'].fillna('C50')
data['Embarked'] = data['Embarked'].fillna('Q')
print(data.isnull().sum())
