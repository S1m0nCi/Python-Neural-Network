import numpy as np

from .NodeChange import NodeChange

class Node:
  def __init__(self, weights: list[float], bias: float, position: list[int] = None):
    self.weights = weights
    self.bias = bias
    self.position = position

  def compute(self, inputs: list[float], activation):
    self.inputs = inputs
    if len(inputs) == len(self.weights):
      self.activation_input = sum([self.weights[i] * inputs[i] for i in range(len(inputs))]) + self.bias
      output = activation(self.activation_input)
      return output
    if len(inputs) > len(self.weights):
      raise RuntimeError("Too many inputs")
    raise RuntimeError("Too few inputs")
  
  # maybe should remove later after refactoring elsewhere: 
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

  def __repr__(self):
    if len(self.weights) > 1:
      return f" Weights: {self.weights} Bias: {self.bias}"
    return f" Weight: {self.weights} Bias: {self.bias}"
    