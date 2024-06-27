import unittest
import numpy as np

from neural_network.classes.Node import Node
from neural_network.classes.NodeChange import NodeChange
from neural_network.functions.activation.sigmoid import sigmoid

class testNodeMethods(unittest.TestCase):

  def testCompute(self):
    node = Node([1, 2], 2)
    result = node.compute([2, 1], sigmoid)
    self.assertEqual(result, 1/(1+np.exp(-6)))

  def testApplyChange(self):
    with self.subTest("Check functionality"):
      node = Node([1, 0.4], 2.4)
      node_change = NodeChange([0.1, 0.2], 0.5)
      node_copy = Node([1, 0.4], 2.4)
      node.applyChange(node_change, 0.9)
      self.assertIsNot(node, node_copy)
    
    with self.subTest("Check correct values"):
      node = Node([2, 1], 1)
      node_change = NodeChange([0.9, 0.5], 0.1)
      node.applyChange(node_change, 1)
      node_check = Node([1.1, 0.5], 0.9)
      self.assertEqual(node.weights, node_check.weights)
      self.assertEqual(node.bias, node_check.bias)
  
if __name__ == "__main__":
  unittest.main()