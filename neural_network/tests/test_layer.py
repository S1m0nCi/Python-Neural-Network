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
      node = Node([1, 1], 0)
      layer = Layer([node])
      layer.compute_layer([[1.0, 1.0], [1.0, 1.0]])
      layer.pass_layer([[0], [1]])

if __name__ == "__main__":
  unittest.main()