import pandas as pd
import math

class Node:
  def __init__(self, feature = None, threshold = None, pre=None, post=None,*, value=None):
    self.feature = feature
    self.threshold = threshold
    self.pre = pre
    self.post = post
    self.value = None

  def is_leaf(self):
    return self.value is not None

  def set_pre(self, pre):
    self.pre = pre
  
  def set_post(self, post):
    self.post = post

class DecisionTree:
  def __init__(self, df, target):
    self.features = []
    self.df = df
    self.target_values = df[target].value_counts
    for feature in df.columns:
      self.features.append(feature)

  def classifier(self):

    entropies = []
    for feature in self.features:
      entropies.append(entropy(feature))

    


  
  def entropy(self, feature):

    for value in self.df[feature].unique():
      l = len(self.df[self.df[feature]==value])
    
    for value in self.target_values:
      entoropy += -(value/l)(math.log2(value/l))

    entropy = 0
    for value in target_values:
      entropy += -(value/sum)(math.log2(value/sum))

    return entropy

    

