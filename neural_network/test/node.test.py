import unittest
import numpy as np

from ..classes.Node import Node
from ..functions.activation.sigmoid import sigmoid

class testNodeMethods(unittest.TestCase):

  def testCompute(self):
    node = Node([1, 2], 2)
    result = node.compute([2, 1], sigmoid)
    self.assertEqual(result, 1/(1+np.exp(-6)))

if __name__ == "__main__":
  unittest.main()