# Mean square error function
def mse(estimated: list, actual: list):
  return sum([(estimated[i] - actual[i])**2 for i in range(len(estimated))])/len(estimated)

def dmse(estimated: list, actual: list, index: int):
  return 2*(estimated[index] - actual[index])/len(estimated)