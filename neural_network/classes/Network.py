import numpy as np

from .Node import Node
from .Layer import Layer
from .NodeChange import NodeChange
from .Path import Path

from ..functions.loss.mse import mse, dmse, dmse1 
from ..functions.activation.sigmoid import sigmoid, dsigmoid
from ..functions.utils import floatify, form_zeros_array

ACTUAL = 0

class Network:
  # we can also have different options for connections, like 'dense', '2', etc
  # layers includes input layer and output layers
  def __init__(self,
               layers: list[int]= [2,2,1],
               node_struct: list[list[int]] = [np.ones(2).tolist(), 0],
               connection: str = "two", 
               activation: str = "sigmoid", 
               loss: str = "mean square error", 
               links: list[list[list[int]]]=[[[0,1], [0,1]], [[0], [0]]], # first layer is input layer
               ):    
    self.shape = layers
    self.nodes = [[Node(node_struct[0], node_struct[1], [j,i]) for i in range(layers[j])] for j in range(len(layers))]
    self.layers = [Layer(self.nodes[i]) for i in range(len(self.nodes))]
    self.layer_results = []
    
    self.links = links
    self.connection = connection
    self.activation = activation
    self.loss = loss
  
  def feed_forward(self, initial: list):
    initial = floatify(initial)
    self.layer_results = []                                                                                                                                                                  
    # create the first input moulding to the input layer
    self.layers[0].result = initial
    current = self.layers[0].pass_layer(self.links[0], self.layers[1].length())
    print (f"current {current}")
    for i in range(1, len(self.layers)-1):
      self.layers[i].compute_layer(current)
      self.layer_results.append(self.layers[i].result)
      current = self.layers[i].pass_layer(self.links[i], self.layers[i+1].length())
      print (f"current {current}")
    self.layers[len(self.layers)-1].compute_layer(current)
    self.final_output = self.layers[len(self.layers)-1].result
    self.layer_results.append(self.final_output)
    return self.final_output
  
  def calculate_loss(self, layer_result, actual_result):
    return mse(layer_result, actual_result)
  
  # this works as a general function to find the path
  def map_path(self, layer_index: int, node_index: int):
    # path has the form of a self.nodes, with the path arranged in layers
    # this function maybe should be recursive. We map the path for each node as we go
    path = [[self.nodes[layer_index][node_index]]]
    current_node_index = node_index
    for i in range(layer_index+1, len(self.layers)):
      next_node_position = self.links[i-1][current_node_index] # will be a list: can be a list of multiple indices
      path.append([self.nodes[i][next_node_position[j]] for j in range(len(next_node_position))])
    return path
  
  def map_paths_recursive(self, layer_index: int, node_index: int, stored_paths: list[list[Path]]):
    path = [[self.nodes[layer_index][node_index]]]
    current_node_index = node_index
    # we do not need a for loop, we need recursion
    # we may need a list of lists similar to self.nodes, but where each node is our path, as a memoisation tool
    # the path is like a tree
    if layer_index == len(self.layers) - 1:
      stored_paths[layer_index][node_index] = Path([path])
      return stored_paths
    next_node_positions = self.links[layer_index][current_node_index]
    next_path_layer = []
    for index in next_node_positions:
      if stored_paths[layer_index+1][index] != 0:
        next_path_layer.append(stored_paths[layer_index+1][index].get_path())
      else:
        next_path_layer.append(self.map_paths_recursive(layer_index+1, index, stored_paths)[layer_index+1][index].get_path())
    path.append(next_path_layer)
    stored_paths[layer_index][node_index] = Path([path])
    return stored_paths
    
  def create_paths(self):
    all_paths = form_zeros_array(self.shape)
    for i in range(len(self.nodes)):
      for j in range(len(self.nodes[i])):
        all_paths[i][j] = self.map_path(i, j) 
    return all_paths
  
  def update(self, learning_rate, correct_output):
    L_components = [dmse(self.layer_results[::-1][0], correct_output, i) for i in range(self.shape[::-1][0])]
    self.node_changes = form_zeros_array(self.shape)
    self.paths = self.create_paths()
    for i in range(len(self.nodes)):
      for j in range(len(self.nodes[i])):
        root_path = self.paths[i][j]
        memo = form_zeros_array([len(lst) for lst in root_path])
        leads_to_output_nodes = root_path[::-1][0]
        weight_output_derivs = [
          [self.sigmoid_chain_w(len(self.shape)-1, m.position[1], root_path, memo, k) for m in leads_to_output_nodes]
          for k in range(len(self.nodes[i][j].weights))
        ]
        bias_output_deriv = [self.sigmoid_chain_b(len(self.shape)-1, m.position[1], root_path, memo) for m in leads_to_output_nodes]
        weight_loss_derivs = [
          L_components[leads_to_output_nodes[k].position[1]]*weight_output_derivs[m][k] for k in range(len(leads_to_output_nodes))
          for m in range(len(weight_output_derivs))
                              ]
        bias_loss_deriv = sum([L_components[leads_to_output_nodes[k].position[1]]*bias_output_deriv[k] for k in range(len(leads_to_output_nodes))])
        self.node_changes[i][j] = NodeChange(weight_loss_derivs, bias_loss_deriv)
        self.nodes[i][j].applyChange(self.node_changes[i][j], learning_rate)


  def map_all_paths(self):
    # all_paths should have the same shape as self.nodes
    all_paths = form_zeros_array(self.shape)
    for i in range(len(self.nodes[0])):
      all_paths = self.map_paths_recursive(0, i, all_paths) #  maybe don't append, do something else instead
    return all_paths
  

  # NOT IN USE
  def compute_node_inputs(self, initial: list): #compute node inputs that are put into activation function
    self.activation_inputs = []
    current = self.layers[0].pass_layer(initial, self.links[0])
    for i in range(1, len(self.layers)-1): # layers, not counting the input layer
      layer_result = self.layers[i].compute_layer(current)

  def backpropagate_and_update(self, learning_rate):
    # assumes 'two' structure
    L_component = dmse(self.layer_results[len(self.layer_results)-1], ACTUAL)
    self.node_changes = form_zeros_array(self.shape) #  same shape
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

  def general_backpropagate(self, learning_rate, correct_output):
    # there should be only one loss function for the output layer that we minimise
    loss = self.calculate_loss(self.layer_results[::-1][0], correct_output)
    print (f"Loss: {loss}")

    self.node_changes = form_zeros_array(self.shape) 
    L_components = [dmse(self.layer_results[::-1][0], correct_output, i) for i in range(self.shape[::-1][0])]
    self.paths = self.map_all_paths()
    for i in range(1, len(self.nodes)): # layers, not counting the input layer
      for j in range(len(self.nodes[i])): # nodes
        root_path = self.paths[i][j]
        # call the sig chain functions to start from the output layer
        leads_to_output_nodes = [path[0].position[1] for path in root_path.get_path()[::-1][0]]
        memo = form_zeros_array([len(lst) for lst in root_path])
        weight_output_derivs = [
          [self.sigmoid_chain_w(len(self.shape)-1, m, root_path, memo, k) for m in leads_to_output_nodes]
          for k in range(len(self.nodes[i][j].weights))
        ]
        bias_output_deriv = [self.sigmoid_chain_b(len(self.shape)-1, m, root_path, memo) for m in leads_to_output_nodes]
        weight_loss_derivs = [
          L_components[leads_to_output_nodes[k]]*weight_output_derivs[m][k] for k in range(len(leads_to_output_nodes))
          for m in range(len(weight_output_derivs))
                              ]
        bias_loss_deriv = [L_components[leads_to_output_nodes[k]]*bias_output_deriv[k] for k in range(len(leads_to_output_nodes))]
        self.node_changes[i][j] = NodeChange(weight_loss_derivs, bias_loss_deriv)
        self.nodes[i][j].applyChange(self.node_changes[i][j], learning_rate)
  
  def sigmoid_chain_w(self, i, j, root_path, path_memo, weight_index):
    # i denotes layer
    # j denotes node index in layer
    # write the self.nodes[i][j] as a product/sum of the derivatives that were before it
    if i == len(self.shape) - len(root_path):
      return dsigmoid(self.nodes[i][j].activation_input)*self.nodes[i][j].inputs[weight_index]
    depends_on = []
    # check layer before:
    for k in range(len(root_path[i-1])):
      if self.nodes[i][j] in self.paths[i-1][root_path[i-1][k].position[1]]:
        depends_on.append(k) # we know the layer is i-1
    # form list of dsigmoids: forming a list allows for easier memoisation implementation later if needed
    depends_on_ds = [
      dsigmoid(self.nodes[i-1][depends_on[k]].activation_input)*self.nodes[i-1][depends_on[k]].weights[k] for k in range(len(depends_on))
    ]
    # form a list of the unknown (or memoised) weight partial derivatives
    depends_on_w = []
    for k in range(len(depends_on)): # we are not looking for the actual index
      if path_memo[i-1][k] != 0:
        depends_on_w.append(path_memo[i-1][k])
      else:
        depends_on_w.append(self.sigmoid_chain_w(i-1, depends_on[k], self.paths[i-1][depends_on[k]], path_memo, weight_index))
    return sum([depends_on_ds[k]*depends_on_w[k] for k in range(len(depends_on))])

  def sigmoid_chain_b(self, i, j, root_path, path_memo):
    # i denotes layer
    # j denotes node index in layer
    if i == len(self.shape) - len(root_path):
      return dsigmoid(self.nodes[i][j].activation_input)
    depends_on = []
    # check layer before:
    for k in range(len(root_path[i-1])):
      if j in self.paths[i-1][root_path[i-1][k].position[1]]:
        depends_on.append(k) # we know the layer is i-1
    # form list of dsigmoids: forming a list allows for easier memoisation implementation later if needed
    depends_on_ds = [
      dsigmoid(self.nodes[i-1][depends_on[k]].activation_input)*self.nodes[i-1][depends_on[k]].weights[k] for k in range(len(depends_on))
    ]
    # form a list of the unknown (or memoised) weight partial derivatives
    depends_on_w = []
    for k in range(len(depends_on)): # we are not looking for the actual index
      if path_memo[i-1][k] != 0:
        depends_on_w.append(path_memo[i-1][k])
      else:
        depends_on_w.append(self.sigmoid_chain_b(i-1, depends_on[k], self.paths[i-1][depends_on[k]], path_memo))
    return sum([depends_on_ds[k]*depends_on_w[k] for k in range(len(depends_on))])
                                                                                        


  # now we deal with the data given to the neural network
  # This is to be used by the developer
  def train(self, learning_rate: float, data: list[list[float]], output_columns: list[int]):
    output_targets = [[data[i][output_column] for output_column in output_columns] for i in range(len(data))]
    for i in range(len(data)):
      self.feed_forward(data[i][:len(data[i])-1])
      output_target = output_targets[i]
      self.calculate_loss(self.layer_results[::-1][0], output_target)
      self.update(learning_rate, output_target)
    final_loss = self.calculate_loss(self.layer_results[::-1][0], output_targets[::-1][0])
    print (f"final loss is {final_loss}")

  def test(self, data: list[list[float]], output_columns: list[int]):
    output_targets = [[data[i][output_column] for output_column in output_columns] for i in range(len(data))]
    test_losses  = []
    for i in range(len(data)):
      self.feed_forward(data[i][:len(data[i])-1])
      output_target = output_targets[i]
      test_losses.append(self.calculate_loss(self.layer_results[::-1][0], output_target))
    avg_loss = sum(test_losses)/len(test_losses)
    print (f"Average loss is {avg_loss}")

  def train_network(self, learning_rate:float, data: list[list[float]], output_columns: list[int]):
    output_targets = [[data[i][output_column] for output_column in output_columns] for i in range(len(data))]
    for i in range(len(data)):
      self.feed_forward(data[i][:len(data[i])-1])
      output_target = output_targets[i]
      self.calculate_loss(self.layer_results[::-1][0], output_target)
      self.general_backpropagate(learning_rate, output_target)
    final_loss = self.calculate_loss(self.layer_results[::-1][0], output_targets[::-1][0])
    return f"final loss is {final_loss}"
  
  # Internal use only:
  def set_nodes(self, nodes):
    self.nodes = nodes
    self.layers = [Layer(self.nodes[i]) for i in range(len(self.nodes))]
    print (self)

  def set_structure(self, nodes, links):
    self.set_nodes(nodes)
    self.change_links(links)

  def change_links(self, links):
    self.links = links
    
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
    

