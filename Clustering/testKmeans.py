from sklearn import datasets
import pandas as pd
from kmeans import KMeans
import matplotlib.pyplot as plt

data = datasets.load_iris()
X = data.data
y = data.target

df = pd.DataFrame(X, columns = data.feature_names)
df.drop(columns=['sepal length (cm)', 'sepal width (cm)'], inplace=True)

# plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
# plt.show()

km = KMeans(k=2)
# km.fit(df[['petal length (cm)', 'petal width (cm)']])