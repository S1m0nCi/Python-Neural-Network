class NodeChange:
  def __init__(self, weight_loss_deriv: list[float], bias_loss_deriv: float):
    self.weights = weight_loss_deriv
    self.bias = bias_loss_deriv