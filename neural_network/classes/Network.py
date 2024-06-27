import numpy as np

from .Node import Node
from .Layer import Layer
from .NodeChange import NodeChange

from ..functions.loss.mse import mse, dmse 
from ..functions.activation.sigmoid import sigmoid, dsigmoid
from ..functions.utils import floatify

ACTUAL = 0

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
               links: list[list[list[int]]]=[[[0,1], [0,1]], [[0], [0]]] # first layer is input layer
               ):
    # the 2 below is specific to the 'two' option
    self.nodes = [[Node(np.ones(2).tolist(), 0) for i in range(layers[j])] for j in range(len(layers))]
    self.layers = [Layer(self.nodes[i]) for i in range(len(self.nodes))]
    self.layer_results = []
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
  def feed_forward(self, initial: list):
    initial = floatify(initial)
    self.layer_results = []                                                                                                                                                                  
    # create the first input moulding to the input layer
    self.layers[0].result = initial
    current = self.layers[0].pass_layer(self.links[0], self.layers[1].length())
    for i in range(1, len(self.layers)-1):
      self.layers[i].compute_layer(current)
      self.layer_results.append(self.layers[i].result)
      current = self.layers[i].pass_layer(self.links[i], self.layers[i+1].length())
    self.layers[len(self.layers)-1].compute_layer(current)
    self.layer_results.append(self.layers[len(self.layers)-1].result)
    return self.layers[len(self.layers)-1].result
    # returns the result for the last node(s)
  
  def calculate_loss(self, layer_result, actual_result):
    # assuming just one node in the output layer
    # using MSE loss
    return mse(layer_result, actual_result)
  
  # assuming 'two'
  def map_path(self, layer_index: int, node_index: int):
    path = [self.nodes[layer_index][node_index]]
    current_node_index = node_index
    for i in range(layer_index+1, len(self.layers)):
      next_node_position = self.links[i-1][current_node_index]
      path.append(self.nodes[i][next_node_position])
    return path
  
  # NOT IN USE
  def compute_node_inputs(self, initial: list): #compute node inputs that are put into activation function
    self.activation_inputs = []
    current = self.layers[0].pass_layer(initial, self.links[0])
    for i in range(1, len(self.layers)-1): # layers, not counting the input layer
      layer_result = self.layers[i].compute_layer(current)

  # we need to store all of the results for all the nodes
  # we also need to save the inputs? or we can just recompute each time - recompute for now
  def backpropagate_and_update(self, learning_rate):
    # assume 'two' structure
    L_component = dmse(self.layer_results[len(self.layer_results)-1], ACTUAL)
    self.node_changes = np.empty(self.layers) #  same shape
    # calculate the necessary partial derivatives for all weights and biases
    # if we just do this for one weight or bias we can just repeat: this is not difficult, after all.
    # first find all the partial derivatives and put them in a list which matches the list of nodes, self.nodes
    # a list(network) of lists(layers): inside each list we could have a new kind of class, or just a weight, weight, bias list again 
    for i in range(1, len(self.nodes)): # layers, not counting the input layer
      for j in range(len(self.nodes[i])): # nodes
        path = self.map_path(i, j)
        # in path we have some nodes: we use their results
        # we need to know which weight is applied along the path
        back_prop = 1
        weight_changes = []
        for k in range(1, len(path)):
          back_prop *= dsigmoid(path[k].compute_activation_input(path[k].inputs))*path[k].weights[path[k].inputs.index(path[k-1].output)]
        for k in range(self.nodes[i][j].weights):
          weight_changes.append(back_prop*L_component*dsigmoid(path[0].compute_activation_input(path[0].inputs))*self.nodes[i][j].inputs[k])
        bias_change = back_prop*L_component*self.nodes[i][j]
        self.node_changes[i][j] = NodeChange(weight_changes, bias_change)
        # then backpropagate the bias as well
        self.nodes[i][j].applyChange(self.node_changes[i][j], learning_rate)

  # now we deal with the data given to the neural network
  # This is to be used by the developer
  def train(self, learning_rate: float, data: list[list[float]]):
    for i in range(len(data)):
      self.feed_forward(data[i][:len(data[i])-1])
      self.calculate_loss(self.layer_results[::-1][0], data[i][::-1][0])
      self.backpropagate_and_update(learning_rate)
    final_loss = self.calculate_loss(self.layer_results[::-1][0], data[i][::-1][0])
    return f"final loss is {final_loss}"

  # To give a visual on the neural network, and also to help with debugging:
  def __str__(self):
    neural_rows = []
    for line in self.nodes:
      last_line_length = 0
      for node in line:
        neural_rows.append(f"***|{node}|***")
        last_line_length += len(f"***|{node}|***")
      neural_rows.append("\n")
      neural_rows.append("*" * last_line_length)
      neural_rows.append("\n")
    return "".join(neural_rows)
    

