import torch
from typing import Optional

import utils.audio_restoration.power_estimation_utils as power_estimation_utils

class BaseConditioner():

    def __init__(self, args, model, diff_params, requires_grad=False):

        self.args = args
        self.model = model.eval() #is it ok to do this here?
        self.diff_params=diff_params
        self.requires_grad = requires_grad

    def get_Tweedie_estimate(self, x, t_i):

        if x.ndim==2:
            x_hat = self.diff_params.denoiser(x.unsqueeze(1), self.model, t_i).squeeze(1)
        elif x.ndim==3:
            x_hat = self.diff_params.denoiser(x, self.model, t_i)

        return x_hat

    def Tweedie2score(self, tweedie, xt, t):
        return self.diff_params.Tweedie2score(tweedie, xt, t)

    def score2Tweedie(self, score, xt, t):
        return self.diff_params.score2Tweedie(score, xt, t)

    def conditioning_term(self, x, x_den, t_i, y, degradation):
        """Return a score correction term with the same shape as x."""
        raise NotImplementedError