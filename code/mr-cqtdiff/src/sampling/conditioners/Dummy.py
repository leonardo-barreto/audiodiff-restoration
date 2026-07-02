import torch

class BaseConditioner:

    requires_grad = False

    def __init__(self, args):
        self.args = args

    def conditioning_term(self, x, x_den, t_i, y, degradation):
        """Return a score correction term with the same shape as x."""
        raise NotImplementedError


class Dummy(BaseConditioner):

    def conditioning_term(self, x, x_den, t_i, y, degradation):
        return torch.zeros_like(x)

