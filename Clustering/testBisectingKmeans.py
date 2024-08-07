import numpy as np
from sklearn import datasets
import pandas as pd
from kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from bisectingKmeans import BisectingKmeans

data = datasets.load_iris()
X = data.data
y = data.target

df = pd.DataFrame(X, columns = data.feature_names)
df.drop(columns=['sepal length (cm)', 'sepal width (cm)'], inplace=True)

# Normalize scale
ms = MinMaxScaler()
df['petal width (cm)'] = ms.fit_transform(df[['petal width (cm)']])
df['petal length (cm)'] = ms.fit_transform(df[['petal length (cm)']])
df['target'] = y
df['cluster'] = -1




# visualize data
# plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
# plt.show()

b_kmeans = BisectingKmeans(k=3)
b_kmeans.bisecting_kmeans(df)
clusters = b_kmeans.clusters

# visualize clusters

df1 = clusters[0]
df2 = clusters[1]
df3 = clusters[2]


plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='red')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='green')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='blue')


# centroids
plt.scatter(b_kmeans.centroids[:,0], b_kmeans.centroids[:,1], marker='*', color='black')

plt.title('Clusters')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()


