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




# visualize data
# plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
# plt.show()

# create k-means model
km = KMeans(k=2)
y_predicted = km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])

# add label column to dataframe
df['cluster'] = y_predicted

# visualize clusters
df1 = df[df['cluster']==0]
df2 =df[df['cluster']==1]


plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='red')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='green')

# centroids
plt.scatter(km.centroids[:,0], km.centroids[:,1], marker='*', color='black')

plt.title('Clusters')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()


