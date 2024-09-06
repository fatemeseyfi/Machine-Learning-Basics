import DT
import pandas as pd
import numpy as np
import math

df = pd.read_csv('D:\\Github\\ML\\Machine-Learning-Basics\\Supervised Learning\\Classification\\weather-nominal-weka (1).csv')


# create an instance
X = pd.DataFrame(df.values[:,0:4], columns=df.columns[0:4])
y = pd.DataFrame(df.values[:,4], columns=[df.columns[4]])



X_test = {
  'outlook': ['rainy', 'overcast', 'sunny'],
  'temperature': ['hot','hot', 'mild'],
  'humidity': ['high', 'normal','high'],
  'windy': [True, True, False]
}
y_test = {
  'play' : ['no', 'no', 'no']
}

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

# create and fit the model
dt = DT.DecisionTree()
dt.fit(X, y)

print(dt.predict(X_test))