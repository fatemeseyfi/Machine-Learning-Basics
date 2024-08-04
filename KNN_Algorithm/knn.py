import numpy as np
from collections import Counter



class KNN:
  def __init__(self,k=3):
    self.k = k
    self.X_train = []
    self.y_train = []
    self.indeces = []

  # we don't do much
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  # predict the label for test set
  def predict(self, X):
    predictions = [self._predict(x) for x in X]
    return predictions

  def _predict(self, x):
    # compute the distances
    distances = [self.euclidean_distance(x,x_train) for x_train in self.X_train]

    # get the closest k
    self.k_indeces = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in self.k_indeces]

    # majority vote
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]
    
# parameters are arrays
  def euclidean_distance(self, u, v):
    return np.sqrt(np.sum((u-v)**2))
