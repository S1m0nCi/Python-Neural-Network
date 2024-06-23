
from Node import Node
from functions.sigmoid import sigmoid

class Layer:
  def __init__(self, nodes: list[Node]):
    self.nodes = nodes
  
  # unsure about below
  def compute_layer(self, inputs: list[list[float]]):
    return [self.nodes[i].compute(inputs[i], sigmoid) for i in range(len(self.nodes))]
  # above function still not in use

  def pass_layer(self, result:list, link:list[list]):
    # return some inputs (initial)
    # just replace the indexes in link with the actual values in result
    return [[result[i] for i in range(len(link[j]))] for j in range(len(link))]

  def length(self):
    return len(self.nodes)
  