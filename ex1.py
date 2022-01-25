import pandas as pd

print('-----------------------')
data = pd.read_csv("train.csv")

men_live = data[(data['Survived'] == 1) & (data['Sex'] == 'male')]['Pclass'].value_counts()
men_dead = data[(data['Survived'] == 0) & (data['Sex'] == 'male')]['Pclass'].value_counts()

women_live = data[(data['Survived'] == 1) & (data['Sex'] == 'female')]['Pclass'].value_counts()
women_dead = data[(data['Survived'] == 0) & (data['Sex'] == 'female')]['Pclass'].value_counts()

print("Количество выживших мужчин:")
print(men_live)
print("Количество погибших мужчин:")
print(men_dead)
print("Количество выживших женщин:")
print(women_live)
print("Количество погибших женщин:")
print(women_dead)
