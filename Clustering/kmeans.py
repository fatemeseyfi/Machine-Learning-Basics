import numpy as np

class KMeans:
  def __init__(self, k=3, max_iter=300):
    self.k = k
    self.max_iter = max_iter
    self.centroids = []
    self.clusters = [[] for _ in self.k]
    self.X = []

  # set cluster for all of datapoints
  def fit(self,X):
    self.X = np.array(X)
    self.set_centroids()

    for _ in range(1,self.max_iter):
      for x in self.X:
        # find the index of closest centroid
        distances = [self.euclidean_distance(x, c) for c in self.centroids]
        index = np.argsort(distances)[0]

        # append the datapoint to proper cluster
        self.clusters[index].append(x)
      
      # at the end of each iteration recalculate centroids
      self.recalculate_centroids()

  # initialize centroids
  def set_centroids(self):
    self.centroids = np.random.random((self.k, self.X.shape[1]))

  # calculate centroids according to the datapoints in each cluster
  def calculate_centroids(self):
    for i in self.k:
      self.centroids[i] = self.clusters[i].mean(axis = 0)

  # calculate euqlidean distance between 2 points
  def euclidean_distance(self, u, v):
    u = np.array(u)
    v = np.array(v)

    return np.sqrt(np.sum((u-v)**2))