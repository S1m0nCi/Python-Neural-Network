import unittest

from neural_network.classes.Network import Network
from neural_network.classes.Node import Node

class testNetwork(unittest.TestCase):
  def testFeedForward(self):
    with self.subTest("Default network feed forward"):
      network = Network() # set up a default network
      network.feed_forward([2, 3])
      print (network.layer_results[::-1][0])


if __name__ == "__main__":
  unittest.main()