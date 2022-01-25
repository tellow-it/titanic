import pandas as pd

print('-----------------------')
data = pd.read_csv("train.csv")

all_people = data['Embarked'].value_counts()
alive = data[(data['Survived'] == 1)]['Embarked'].value_counts()
print(all_people)
print(alive)

print(alive / all_people)
print("Мы видим, что людей с порта S выжило больше, "
      "чем с остальных портов, но в процентном соотношении с порта С выжило больше, чем с других")
