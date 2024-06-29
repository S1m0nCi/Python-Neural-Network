import numpy as np

# makes a list of integers a list of floats
def floatify(lst):
  return [float(num) for num in lst]

# makes an zeros array (list of lists) that fits a shape with non-constant 'width'
def form_zeros_array(shape_lst):
  output_lst = []
  for i in range(shape_lst):
    output_lst.append(np.zeros(shape_lst[i]).tolist())
  