from tqdm import tqdm
import torch

class Dummy():

    #def __init__(self, args):

    def conditioning_term(self, x, x_den, y, t_i, degradation):
        return 0.0

class BaseConditioner():

    def __init__(self, args):
        pass

    def conditioning_term(self, x, x_den, t_i):
        pass

    def step_callback(self, sampler_state):
        pass