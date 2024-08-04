from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from base import KNN
from IB2 import IB2
from IB3 import IB3
import numpy as np
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# load iris dataset
iris = datasets.load_iris()

# create train and test set
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Visualize dataset
# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# create knn model
knn_classifier = KNN(k=5)
knn_classifier.fit(X_train, y_train)
pred1 = knn_classifier.predict(X_test)

# create IB2 model
ib2_classifier = IB2(k=5)
ib2_classifier.fit(X_train, y_train)
pred2 = ib2_classifier.predict(X_test)

# create IB3 model
ib3_classifier = IB3(k=5)
ib3_classifier.fit(X_train, y_train)
pred3 = ib3_classifier.predict(X_test)

# accuracy
acc1 = np.sum(pred1 == y_test)/len(y_test)
acc2 = np.sum(pred2 == y_test)/len(y_test)
acc3 = np.sum(pred3 == y_test)/len(y_test)

print(f"acc of knn: {acc1}   acc of ib2: {acc2}   acc of ib3: {acc3}")
