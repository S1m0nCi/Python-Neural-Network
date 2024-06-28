import numpy as np
from math import exp

def sigmoid(x):
  return 1/(1+exp(-x))

# we will also need the derivative of the sigmoid function
# OR we could approximate the derivative: using first principles

def dsigmoid(x):
  return exp(-x)/((1+exp(-x))**2)