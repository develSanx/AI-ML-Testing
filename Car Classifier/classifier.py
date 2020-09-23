import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')
le = preprocessing.LabelEncoder()
"""
using LabelEncoder obj transforms non int values to int
"""
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

"""
KNeighbor algorithm needs calculation always. so it's not woth training and storing
"""

print("Car Classifier\n")
buying = input("Enter Buying: ")
maint = input("Enter maint: ")
door = input("Enter door: ")
persons = input("Enter persons: ")
lug_boot = input("Enter lug_boot: ")
safety = input("Enter safety: ")

buying = le.fit_transform(list(buying))
maint = le.fit_transform(list(maint))
door = le.fit_transform(list(door))
persons = le.fit_transform(list(persons))
lug_boot = le.fit_transform(list(lug_boot))
safety = le.fit_transform(list(safety))

inp_data = list(zip(buying, maint, door, persons, lug_boot, safety))
predicted = model.predict(inp_data)
names = ['unacc', 'acc', 'good', 'vgood']

print("\nClass might be ", names[predicted[0]])

