from knn import KNN
class IB2(KNN):
  def __init__(self,k):
    super().__init__(k)

  def fit(self, X, y):
    for x, label in zip(X, y):
      if not self.X_train: # if X_trian is empty
        self.X_train.append(x)
        self.y_train.append(label)
      else:
        pred = self._predict(x)
        if (pred != label):
          self.X_train.append(x)
          self.y_train.append(label)