import pandas as pd
import numpy as np
import math

class Node:
  def __init__(self, feature = None, pre_node=None, in_edge = None, edges=None, value=None):
    self.feature = feature
    self.pre_node = pre_node
    self.in_edge = in_edge
    self.value = None
    self.edges = edges

  


class DecisionTree:
  def __init__(self):
    self.features = []
    self.nodes = []

  # recursive function
  def create_tree(self, node, sub_df):

    # if features list is not empty
    if self.features:
      self.set_leaf(node, sub_df)

      # if node is not a leaf
      if not node.value:
        # find the smallest avg entropy
        entropies = []
        for feature in self.features:
          entropies.append(self.entropy(feature))

        # removes the feature with smallest entropy
        index = np.argsort(entropies)[0]
        label = self.features.pop(index)

        # adds that node to the tree
        node.feature = label
        node.edges = sub_df[label].unique()
        self.nodes.append(node)

        # recall the function for the values of that feature
        for value in node.edges:
          next_node = Node(pre_node = node, in_edge=value)
          self.create_tree(next_node, sub_df[sub_df[node.feature]==value])
      
      # if the node is a leaf
      else:
        # adds that node to the tree and return
        self.nodes.append(node)
        return

  # fits the data
  def fit(self, X, y):
    node = Node()
    for feature in X.columns:
      self.features.append(feature)
    self.target_values = y.iloc[0].unique()
    self.target = y.columns[0]
    self.df = pd.concat([X,y],axis=1)

    # creates tree with data
    self.create_tree(node, self.df)
  
  # calculate avg entropy for a feature
  def entropy(self, feature):

    avg_entropy = 0
    len_df = len(self.df)

    for value in self.df[feature].unique():
      temp_df = self.df[self.df[feature]==value]
      l = len(temp_df)
  
      entropy =0
      for tv in self.target_values:
        count = len(temp_df[temp_df[self.target] == tv])
        probability = count / l

        if probability > 0:  
          entropy += -probability * math.log2(probability)

      avg_entropy += (l/len_df) * entropy

    return avg_entropy

  def set_leaf(self, node, sub_df):

    leaf = None

    if len(sub_df[self.target].unique()) == 1:
      leaf = sub_df[self.target].unique()
  
    node.value = leaf

  def predict(self, X):
    y_pred = []
    for i in range(len(X)): 
      temp_nodes = self.nodes[:] 
      index_to_pop = next((i for i, x in enumerate(temp_nodes) if x.pre_node is None), None)
      node = temp_nodes.pop(index_to_pop)
      y_pred.append(self.find_label(X.iloc[[i]], node, temp_nodes))  
    return y_pred

  def find_label(self, X, node, temp_nodes):

    if not node.value:
      edge = X[node.feature].unique()
      index_to_pop = next(
        (i for i, x in enumerate(temp_nodes) if x.in_edge == edge and x.pre_node == node), 
        None)    
      next_node = temp_nodes.pop(index_to_pop)
      return self.find_label(X, next_node, temp_nodes)

    else:
      return node.value
    
    

