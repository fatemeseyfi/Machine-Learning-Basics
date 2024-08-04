from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from from_scratch import KNN
import numpy as np
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# load iris dataset
iris = datasets.load_iris()

# create train and test set
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Visualize dataset
plt.figure()
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# create knn model
knn_classifier = KNN(k=3)
knn_classifier.fit(X_train, y_train)
pred = knn_classifier.predict(X_test)
# print(pred)

# accuracy
acc = np.sum(pred == y_test)/len(y_test)
print(acc)
