import unittest
import numpy as np

from neural_network.classes.Network import Network
from neural_network.classes.Node import Node

class testNetwork(unittest.TestCase):
  def testFeedForward(self):
    with self.subTest("Default network feed forward"):
      network = Network() # set up a default network
      network.feed_forward([0, 0])
      print (network.final_output)
      #self.assertEqual()

    with self.subTest("Simple feed forward, check result"):
      network = Network()
      network.set_nodes([[Node([0, 1], 0) for i in range(network.layers[j].length())] for j in range(len(network.layers))])
      final_result = network.feed_forward([2, 3])[0]
      self.assertAlmostEqual(final_result, 0.7216, places=4)

    with self.subTest("2-1 feed forward"):
      network = Network(layers=[8,4,2,1])
      final_result = network.feed_forward([0, 1, 0, 1])[0]
      print (final_result)
      


if __name__ == "__main__":
  unittest.main()