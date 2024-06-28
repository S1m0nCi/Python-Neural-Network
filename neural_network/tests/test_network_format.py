import unittest

from neural_network.classes.NetworkFormat import NetworkFormat 

class TestNetworkFormat(unittest.TestCase):
  
  def testTwoCreationSuccess(self):
    with self.subTest("Initial layer 8"):
      networkFormat = NetworkFormat("two")
      links = networkFormat.create_two_link([8, 4, 2, 1])
      self.assertListEqual(links, [[[0], [0], [1], [1], [2], [2], [3], [3]], [[0], [0], [1], [1]], [[0], [0]]])

    with self.subTest("Initial layer 2"):
      networkFormat = NetworkFormat("two")
      links = networkFormat.create_two_link([2, 1])
      self.assertListEqual(links, [[[0], [0]]])

    with self.subTest("Initial layer 32"):
      networkFormat = NetworkFormat("two")
      links = networkFormat.create_two_link([32, 16, 8, 4, 2, 1])
      self.assertListEqual(links, [[[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6], [7], [7], 
                                    [8], [8], [9], [9], [10], [10], [11], [11], [12], [12], [13], [13], [14], [14], [15], [15]],
                                   [[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6], [7], [7]], 
                                   [[0], [0], [1], [1], [2], [2], [3], [3]], 
                                   [[0], [0], [1], [1]], 
                                   [[0], [0]]])
  
  def testTwoCreationError(self):
    with self.subTest("Creating an error"):
      networkFormat = NetworkFormat("two")
      try:
        links = networkFormat.create_two_link([7, 4, 2, 1])
        exception = None
      except Exception as e:
        exception = e
      finally:
        self.assertIsInstance(exception, RuntimeError)

if __name__ == "__main__":
  unittest.main()