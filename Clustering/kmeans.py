import numpy as np

class KMeans:
  def __init__(self, k=3, max_iter=300):
    
    self.k = k
    self.max_iter = max_iter
    self.centroids = np.array([])
    self.clusters = [[] for _ in range(self.k)]
    self.X = np.array([])

  # set cluster for all of datapoints
  def fit_predict(self,X):
    self.X = np.array(X)
    self.set_centroids()
    y_predicted = []

    for _ in range(1,self.max_iter):
      self.clusters = [[] for _ in range(self.k)]
      y_predicted = []

      for x in self.X:
        # find the index of closest centroid
        distances = [self.euclidean_distance(x, c) for c in self.centroids]
        index = np.argsort(distances)[0]

        # append the datapoint to proper cluster
        self.clusters[index].append(x)
        y_predicted.append(index)
      
      # at the end of each iteration recalculate centroids
      self.calculate_centroids()

    return y_predicted

  # initialize centroids
  def set_centroids(self):
    self.centroids = np.random.random((self.k, self.X.shape[1]))

  # calculate centroids according to the datapoints in each cluster
  def calculate_centroids(self):
    for i in range(self.k):
      if self.clusters[i]:  # Avoid division by zero
        self.centroids[i] = np.mean(self.clusters[i], axis=0)

  # calculate euqlidean distance between 2 points
  def euclidean_distance(self, u, v):
    u = np.array(u)
    v = np.array(v)

    return np.sqrt(np.sum((u-v)**2))