import numpy as np

class Node:
  def __init__(self, weights: list[float], bias: float):
    self.weights = weights
    self.bias = bias

  def compute(self, inputs: list[float], activation: function):
    if len(inputs) == len(self.weights):
      return activation(sum([self.weights[i] * inputs[i] for i in range(len(inputs))]) + self.bias)
    if len(inputs) > len(self.weights):
      raise RuntimeError("Too many inputs")
    raise RuntimeError("Too few inputs")
  
  def update(self, weights: list[float], bias: float):
    self.weights = weights
    self.bias = bias