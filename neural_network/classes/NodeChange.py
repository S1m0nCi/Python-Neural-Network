class NodeChange:
  def __init__(self, weight_changes: list[float], bias_change: float):
    self.weights = weight_changes
    self.bias = bias_change