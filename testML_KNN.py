import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("car.data")

# for converting non numeric data into a numeric
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
model = KNeighborsClassifier(n_neighbors=6)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
#
"""
# shows the average probabilities for different K values 
avg_list = list()
for i in range(1, 26):
    avg = 0
    model = KNeighborsClassifier(n_neighbors=i)
    for j in range(1, 51):
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        avg = ((avg*(j-1)) + acc)/j
    print("K =", i, " Accuracy:", avg)
    avg_list.append(avg)

style.use("ggplot")
pyplot.scatter(range(1, 26), avg_list)
pyplot.xlabel("K")
pyplot.xlabel("Probability")
pyplot.show()
"""

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(predicted)):
    print("Predicted:", names[predicted[i]], "Data:", x_test[i], "Actual:", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 4, True)
    print("N =", n)
