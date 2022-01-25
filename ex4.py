import pandas as pd

print('-----------------------')
data = pd.read_csv("train.csv")

name = []
names = []
b = data.Name
for i in b:
    y = i.split(',')[0]
    name.append(y)
for i in name:
    k = name.count(i)
    names.append([i, k])
    while i in name:
        name.remove(i)
l = len(names)
names = sorted(names, key=lambda names: names[1])
top_names = names[l - 11: l]
for i in range(len(top_names) - 1, 0, -1):
    print(str(top_names[i][0]) + "\t" + str(top_names[i][1]) + "\t" + "times")
