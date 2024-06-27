import numpy as np

from .Node import Node
from ..functions.activation.sigmoid import sigmoid

class Layer:
  def __init__(self, nodes: list[Node]):
    self.nodes = nodes
  
  # unsure about below
  def compute_layer(self, inputs: list[list[float]]):
    self.result = [self.nodes[i].compute(inputs[i], sigmoid) for i in range(len(self.nodes))]
    return self.result
  # above function still not in use
  
  # passes to the next layer: formats inputs for the next layer of nodes
  def pass_layer(self, link:list[list], next_layer_len: int):
    # return some inputs (initial)
    # just replace the indexes in link with the actual values in result: not if n to m, m < n!
    next_layer_input = np.empty((next_layer_len, 0)).tolist() # needs to be a list of lists
    for j in range(len(link)):
      for i in range(len(link[j])):
        next_layer_input[link[j][i]].append(self.result[i])
    return next_layer_input
    # now an issue is, are we getting the right order? are the new inputs matching to the right weights? maybe doesn't even matter actually:
    # as long as the order stays the same throughout training
    # return [[self.result[i] for i in range(len(link[j]))] for j in range(len(link))]

  def length(self):
    return len(self.nodes)
   
  def compute_layer_activation_inputs(self, inputs):
    return [self.nodes[i].compute_activation_input(inputs[i]) for i in range(len(self.nodes))]

  