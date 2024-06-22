import numpy as np

from Node import Node
from Layer import Layer

class Network:
  # each layer should be a list of nodes
  # how do we work out how the nodes are connected
  # maybe we should leave this to personalisation ie to the user's choice: give the user fine-tune control
  # we can also have different options for connections, like 'dense', '2', etc
  # layers includes input layer and output layers
  def __init__(self,
               layers: list[int]= [2,2,1],
               connection: str = "two", 
               activation: str = "sigmoid", 
               loss: str = "mean square error", 
               links=[[[0,1], [0,1]], [[0], [0]]]
               ):
    # the 2 below is specific to the 'two' option
    self.nodes = [[Node(np.ones(2).tolist(), 0) for i in range(len(layers[j]))] for j in range(len(layers))]
    self.layers = [Layer(self.nodes[i]) for i in range(len(self.nodes))]
    # the links between layers should be below
    # for now, have simple halving links
    # the output of each layer will go to a layer with half as many nodes
    self.links = links
    self.connection = connection
    self.activation = activation
    self.loss = loss
  
  # for now, we will define all functions using the default settings, just for now
  # then we will make a 'train' function bringing everything together
  # start by computing for the input layer

  # can we use recursion?
  # repeatedly compute layer to get the final output
  def feed_forward(self, initial):
    # create the first input moulding to the input layer
    current = self.layers[0].pass_layer(initial, self.links[0])
    for i in range(1, len(self.layers)-1):
      layer_result = self.layers[i].compute_layer(current)
      current = self.layers[i].pass_layer(self.links[i])
    layer_result = self.layers[len(self.layers)-1].compute_layer(current)
    return layer_result
    # returns the result for the last node(s)

  
