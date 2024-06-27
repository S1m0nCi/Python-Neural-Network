import unittest
import numpy as np

from neural_network.classes.Layer import Layer
from neural_network.classes.Node import Node

from neural_network.functions.activation.sigmoid import sigmoid

class testLayerMethods(unittest.TestCase):
  
  def testCompute(self):
    with self.subTest("Check values (singular)"):
      node = Node([1, 1], 1)
      layer = Layer([node])
      layer.compute_layer([[1.0, 1.0]])
      self.assertListEqual(layer.result, [node.compute([1.0, 1.0], sigmoid)])

  def testPassLayer(self):
    with self.subTest("Simple layer passing"):
      node1 = Node([1, 1], 0)
      node2 = Node([1, 1], 0)
      layer = Layer([node1, node2])
      layer.compute_layer([np.ones(2).tolist(), np.ones(2).tolist()])
      next_input = layer.pass_layer([[0], [1]], 2)
      exp_output = node1.compute(np.ones(2).tolist(), sigmoid)
      self.assertListEqual(next_input, [[exp_output], [exp_output]])
    
    with self.subTest("2-1 layer passing"):
      node1 = Node([1, 1], 0)
      node2 = Node([1, 1], 0)
      layer.compute_layer([np.ones(2).tolist(), np.ones(2).tolist()])
      layer = Layer([node1, node2])
      

if __name__ == "__main__":
  unittest.main()