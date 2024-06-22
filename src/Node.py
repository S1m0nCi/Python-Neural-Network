import numpy as np

class Node:
  def __init__(self, weights: list[float], bias: float):
    self.weights = weights
    self.bias = bias

  def feed_forward(self, inputs):
    if len(inputs) == len(self.weights):
      return sum([self.weights[i] * inputs[i] for i in range(len(inputs))]) + self.bias
    if len(inputs) > len(self.weights):
      raise RuntimeError("Too many inputs")
    raise RuntimeError("Too few inputs")