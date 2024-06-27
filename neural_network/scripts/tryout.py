import numpy as np

from neural_network.classes.Network import Network

network = Network()

print (network)

"""
print (np.ones(2))
print (type(np.ones(2)))
print (np.ones(2).tolist())
print (type(np.ones(2).tolist()))
print (type([1,2]))
"""
empty_array = np.empty((2,0)).tolist()
print(empty_array)

"""
a = 3
b = a
a += 1
print (a)
print (b)

"""

a = [1, 2]
b = a
a.append(3)
print (a)
print (b)
print (np.ones(2).tolist())
