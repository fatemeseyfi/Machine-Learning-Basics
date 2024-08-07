import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

class BisectingKmeans:
  def __init__(self,k=2, max_iter=100):
    self.k = k
    self.clusters = []
    self.max_iter = max_iter
    self.centroids = []


  def get_cluster(self):
    sse = []
    for cluster, centroid in zip(self.clusters, self.centroids):
        cluster_array = cluster.to_numpy()[:,:2]
        squared_diffs = np.sum((cluster_array - centroid) ** 2, axis=1)
        # Sum up the squared differences
        cluster_sse = np.sum(squared_diffs)
        sse.append(cluster_sse)
    
    # Get the index of the cluster with the highest SSE
    c_index = np.argmax(sse)
    return c_index, self.clusters[c_index]

  def bisecting_kmeans(self,df):
    km = KMeans(n_clusters=1)
    km.fit(df[['petal length (cm)', 'petal width (cm)']])
    self.centroids = km.cluster_centers_
    self.clusters = [df]


    # until the number of clusters are k
    while len(self.clusters) < self.k:

      # get a cluster from list 
      i, cluster = self.get_cluster()
      # and remove it
      self.clusters.pop(i)
     
      self.centroids = np.delete(self.centroids, i, axis=0)
    
      # make 2 clusters with that one
      km = KMeans(n_clusters = 2, max_iter = 300)
      y_predicted = km.fit_predict(cluster[['petal length (cm)', 'petal width (cm)']])
      cluster['cluster'] = y_predicted

      # update other clusters label
      self.update_labels()

      # add generated clusters to the list
      df1 = cluster[cluster['cluster'] == 0]
      df2 = cluster[cluster['cluster'] == 1]
      self.clusters.append(df1)
      self.clusters.append(df2)

      # update centroids
      c1 = km.cluster_centers_[0]
      c2 = km.cluster_centers_[1]
      c1 = c1[np.newaxis, :]
      c2 = c2[np.newaxis, :]
      self.centroids = np.append(self.centroids, c1, axis=0)
      self.centroids = np.append(self.centroids, c2, axis=0)

  def update_labels(self):
    for i, cluster in enumerate(self.clusters):
      cluster['cluster'] += i + 1
