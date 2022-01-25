import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import *

print('-----------------------')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('Размер обучающих данных: {}'.format(train.shape))
print('Масштаб данных тестового набора: {}'.format(test.shape))
# При установке параметра ignore_index = True объединенный набор данных восстановит индекс
data = pd.concat([train, test], ignore_index=True)
data['Age'] = data['Age'].fillna((data['Age'].median()))
data['Fare'] = data['Fare'].fillna((data['Fare'].median()))
data['Cabin'] = data['Cabin'].fillna('C50')
# Извлечение функций Как узнать, какие функции важны? Обычно необходимо общаться с людьми, которые знакомы с
# бизнес-логикой, отражают характеристики бизнес-персонала в коде и продолжают пытаться с помощью экспериментов и
# опыта создавать новые функции.

# Пол:
sex_dict = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_dict)
# Посадка (порт посадки): Горячее кодирование с использованием get_dummies Используйте get_dummies для однократного
# кодирования для создания фиктивных переменных, префикс имени столбца (префикс) запускается
EmbarkedDf = pd.get_dummies(data['Embarked'], prefix='Embarked')
# Добавление функций EmbarkedDf в полный набор данных
data = pd.concat([data, EmbarkedDf], axis=1)  # axis = 1 означает вставку данных по столбцу
# Поскольку запущенный порт (Embarked) использовался для горячего кодирования для генерации фиктивных переменных (
# фиктивных переменных), Таким образом, отправленный порт (Embarked) удаляется здесь
data = data.drop('Embarked', axis=1)
# Pclass (класс кабины)
# То же, что и выше
PclassDf = pd.get_dummies(data['Pclass'], prefix='Pclass')
data = pd.concat([data, PclassDf], axis=1)
data = data.drop('Pclass', axis=1)


# Имя (имя пассажира): Из приведенной выше строки «Имя» обнаруживается, что каждое имя содержит заголовок. Мы можем
# получить заголовок каждого пассажира, что может помочь нам проанализировать более полезную информацию.
def getTitle(name):
    s1 = name.split(',')[1]
    s2 = s1.split('.')[0]
    return s2.strip()  # Удалить пробелы в начале и конце строк


data['Title'] = data['Name'].map(getTitle)
data['Title'].value_counts()

# Соответствуют вышеперечисленным названиям в следующих категориях:
#
# Должностное лицо: государственный служащий;
# Роялти: королевская семья (королевская семья);
# Г-н: женатый мужчина;
# миссис: замужняя женщина;
# Мисс: молодая незамужняя женщина;
# Мастер: квалифицированный человек / учитель

title_dict = {"Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty", "Don": "Royalty",
              "Sir": "Royalty", "Dr": "Officer", "Rev": "Officer", "the Countess": "Royalty", "Dona": "Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr", "Mrs": "Mrs",
              "Miss": "Miss", "Master": "Master", "Lady": "Royalty"}
data['Title'] = data['Title'].map(title_dict)
# Горячее кодирование с использованием указанного выше кадра данных заголовка

TitleDf = pd.get_dummies(data['Title'])  # Горячее кодирование
data = pd.concat([data, TitleDf], axis=1)  # Добавить функцию в исходный набор данных
data = data.drop(['Name', 'Title'], axis=1)  # Удалить ненужные столбцы
# Салон титаника:
# Значение категории номера гостя - это первая буква, поэтому мы извлекаем первую букву номера кабины как функцию.

data['Cabin'] = data['Cabin'].map(lambda x: x[0])
data['Cabin'].value_counts()

CabinDf = pd.get_dummies(data['Cabin'], prefix='Cabin')
data = pd.concat([data, CabinDf], axis=1)
data = data.drop('Cabin', axis=1)

# Установить размер семьи и тип семьи:
# Количество семей = количество ближайших родственников одного поколения (Parch)
# + количество ближайших родственников разных поколений (SibSp) + сами пассажиры (поскольку пассажиры также являются
# одним из членов семьи, добавьте 1 здесь)
#
# Family_Single: количество семей = 1
#
# Family_Small: 2 <= количество семей <= 4
#
# Family_Large: Количество семей> = 5

data['familysize'] = data['Parch'] + data['SibSp'] + 1
data['family_singel'] = np.where(data['familysize'] == 1, 1, 0)
data['family_small'] = np.where((data['familysize'] >= 2) & (data['familysize'] <= 4), 1, 0)
data['family_large'] = np.where(data['familysize'] >= 5, 1, 0)

# Выбор характеристик и уменьшение их размеров Благодаря предыдущему выбору объекта, получить 32 объекта,
# следующий использует метод коэффициента корреляции для выбора объектов

# Расчет корреляционной матрицы
corr_df = data.corr()
# Извлечь коэффициенты корреляции каждого признака и Выжившего и расположить их в порядке убывания
corr_df['Survived'].sort_values(ascending=False)
# В соответствии с коэффициентом корреляции каждой функции и Survived для моделирования выбраны следующие функции:
# title (TitleDf в предыдущем наборе данных), класс кабины (PclassDf), цена билета (Fare), номер кабины (CabinDf),
# Порт посадки (EmbarkedDf), пол (пол), размер и категория семьи (familysize, family_small, family_large,
# family_singel)

data_x = pd.concat(
    [TitleDf, PclassDf, CabinDf, EmbarkedDf, data['Fare'], data['Sex'], data['familysize'], data['family_small']
        , data['family_large'], data['family_singel']], axis=1)
# Построить модель
# Установить набор данных для обучения и тестовый набор данных
# Согласно предыдущим данным, мы знаем,
# что train.csv содержит теги Survived, поэтому он используется в качестве обучающих данных модели и должен быть
# разделен на наборы обучающих данных и наборов тестовых данных. Test.csv не имеет тегов Survived и используется в
# качестве данных прогноза устанавливать
#   891 Поведение оригинальных тренировочных данных, мы их извлекаем
source_x = data_x.loc[0:890, :]  # Извлечь особенности
source_y = data.loc[0:890, 'Survived']  # Извлечь теги
#   418 Данные прогнозирования поведения
pred_x = data_x.loc[891:, :]

# Набор обучающих данных и набор тестовых данных, используемых для построения модели, разделенный на тренировочные
# данные и тестовые данные в соответствии с 28-м принципом, 80% из которых являются тренировочными данными

train_x, test_x, train_y, test_y = train_test_split(source_x, source_y, train_size=0.8)
# Функции набора обучающих данных: (712, 27), метка набора обучающих данных: (712,) Характеристики набора тестовых
# данных: (179, 27), метка набора тестовых данных: (179,)
# Standardize train_x, test_x

sc = StandardScaler()
train_x_std = sc.fit_transform(train_x)
test_x_std = sc.transform(test_x)
# Шаг 1: Выберите алгоритм и импортируйте соответствующий пакет вычислений
# Шаг 2. Создание модели
model = LogisticRegression()
# Шаг 3: тренировка модели
model.fit(train_x_std, train_y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                   penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)
model.score(test_x_std, test_y)

# Используйте обученную модель, чтобы предсказать выживание pred_x
pred_x_std = sc.fit_transform(pred_x)
pred_y = model.predict(pred_x_std)

pred_df = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred_y})
pred_df['Survived'] = pred_df['Survived'].astype('int')
pred_df.to_csv('predict.csv', index=False)
