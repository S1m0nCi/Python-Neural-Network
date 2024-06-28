from math import floor
import numpy as np 

# This class is to be used to create links for the different types of network:
# - dense, two, partial dense etc



class NetworkFormat():
  def __init__(self, structure_name):
    self.structure = structure_name

  def create_two_link(self, shape):
    # first check that the shape is correct
    for i in range(len(shape)-1):
      try:
        assert shape[i]==shape[i+1]*2
      except:
        raise RuntimeError("layers are not correct for 'two' format")
    links = np.empty((len(shape)-1, 0)).tolist()
    for i in range(len(shape)-1):
      for j in range(shape[i+1]):
        links[i] += [[j], [j]]
    self.links = links
    return links



    
    

