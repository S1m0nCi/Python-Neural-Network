
from Node import Node

class Layer:
  def __init__(self, nodes):
    self.nodes = nodes

  def compute_layer(inputs: list[list[float]], layer: list[Node]):
    return [layer[i].compute(inputs[i]) for i in range(len(layer))]
  