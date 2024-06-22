import numpy as np

from Node import Node
from Layer import Layer

class Network:
  # each layer should be a list of nodes
  # how do we work out how the nodes are connected
  # maybe we should leave this to personalisation ie to the user's choice: give the user fine-tune control
  # we can also have different options for connections, like 'dense', '2', etc
  def __init__(self, layers: list[int], connection: str = "two", activation: str = "sigmoid", loss: str = "mean square error"):
    self.layers = layers
    self.nodes = [[Node(np.ones(2).tolist(), 0) for i in range(len(layers[j]))] for j in range(len(layers))]
    self.connection = connection
    self.activation = activation
    self.loss = loss
  
  # for now, we will define all functions using the default settings, just for now
  # then we will make a 'train' function bringing everything together
  # start by computing for the input layer
  def compute_layer(inputs: list[list[float]], layer: list[Node]):
    return [layer[i].compute(inputs[i]) for i in range(len(layer))]

  def feed_forward(self):
    initial_inputs = []
    # inputs = compute_layer(initial_inputs, )
