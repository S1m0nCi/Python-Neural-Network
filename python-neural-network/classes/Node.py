import numpy as np

from NodeChange import NodeChange

class Node:
  def __init__(self, weights: list[float], bias: float):
    self.weights = weights
    self.bias = bias

  def compute(self, inputs: list[float], activation: function):
    self.inputs = inputs
    if len(inputs) == len(self.weights):
      output = activation(sum([self.weights[i] * inputs[i] for i in range(len(inputs))]) + self.bias)
      self.output = output
      return output
    if len(inputs) > len(self.weights):
      raise RuntimeError("Too many inputs")
    raise RuntimeError("Too few inputs")
  
  def compute_activation_input(self, inputs: list[float]):
    if len(inputs) == len(self.weights):
      return sum([self.weights[i] * inputs[i] for i in range(len(inputs))]) + self.bias
    if len(inputs) > len(self.weights):
      raise RuntimeError("Too many inputs")
    raise RuntimeError("Too few inputs")
  
  def updateNode(self, weights: list[float], bias: float):
    self.weights = weights
    self.bias = bias

  def applyChange(self, nodeChange: NodeChange, learning_rate: float):
    new_weights = [self.weights[i] - learning_rate*nodeChange.weights[i] for i in range(len(self.weights))]
    new_bias = self.bias - learning_rate*nodeChange.bias
    self.updateNode(new_weights, new_bias)
