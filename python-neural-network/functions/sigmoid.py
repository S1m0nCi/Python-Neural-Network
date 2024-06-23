import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(x))

# we will also need the derivative of the sigmoid function
# OR we could approximate the derivative: using first principles