from knn import KNN
import numpy as np
class IB3(KNN):
  def __init__(self,k=3):
    super().__init__(k)
    # indicators is a list of lists, first item of lists shows the number of samples which are
    # classified correctly with that sample, and the second item shows the number of samples which
    # are calssified incorrectly with that sample
    self.indicators = []
    self.temp_x = []
    self.temp_y = []

  def fit(self, X, y):
    true_th = 1
    false_th = 0

    for x, label in zip(X,y):

      if not self.temp_x:
        self.temp_x.append(x)
        self.temp_y.append(label)
        self.indicators.append([0,0])
      
      else:
        self.temp_x.append(x)
        self.temp_y.append(label)
        self.indicators.append([0,0])

        # find the closest train sample and its label
        index, pred_label = self.find_closest(x)

        # adjust the indicator
        if(pred_label == label):
          self.indicators[index][0] += 1
        else:
          self.indicators[index][1] += 1


    # keep the proper ones for X_train set
    for indicator, x, label in zip(self.indicators, self.temp_x, self.temp_y):
      if indicator[0] >= true_th and indicator[1] <= true_th:
        self.X_train.append(x)
        self.y_train.append(label)


  # find the closest train sample and its label
  def find_closest(self,x):
    distances = [self.euclidean_distance(x, x_train) for x_train in self.temp_x]
    
    x_index = np.argmin(distances)
    label = self.temp_y[x_index]

    return x_index, label

  def print_indicators(self):
    for x in self.indicators:
      print(x, end="\n")
