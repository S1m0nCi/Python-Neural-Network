import unittest
import numpy as np

from neural_network.classes.Network import Network
from neural_network.classes.Node import Node



network = Network()
network.set_nodes([[Node([0, 1], 0) for i in range(network.layers[j].length())] for j in range(len(network.layers))])
final_result = network.feed_forward([2, 3])
print (final_result)