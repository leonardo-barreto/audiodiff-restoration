import torch
from typing import Optional

from utils.linalg import conjugate_gradient, gmres

import utils.audio_restoration.power_estimation_utils as power_estimation_utils
from conditioners.BaseConditioner import BaseConditioner

class MomentMatching(BaseConditioner):

    def __init__(self, args, model, diff_params, maxiter=1, infer_xi=None, xi=None, max_thresh_grads=None, requires_grad=True):
        super().__init__(args, model, diff_params, requires_grad)

        #self.infer_xi = self.args.conditioner.infer_xi if infer_xi is None else infer_xi
        self.infer_xi = infer_xi
        #self.threshold_on_grads = self.args.conditioner.max_thresh_grads if max_thresh_grads is None else max_thresh_grads
        self.threshold_on_grads = max_thresh_grads
        print(f"Conditioner: MomentMatching with infer_xi={self.infer_xi} and threshold_on_grads={self.threshold_on_grads}")
        
        self.maxiter = maxiter

        if self.infer_xi == True:
            floor, ceil = power_estimation_utils.get_floor_and_ceil_powers(self.args)
            self.xi = power_estimation_utils.get_xi_estimate(floor)
            print(f"XI Estimate: {self.xi}")
        else:
            #self.xi = self.args.conditioner.xi if xi is None else xi
            self.xi = xi
            print(f"XI Set: {self.xi}")

    # TO AGENT: THIS FUNCTION WAS BROUGHT FROM MMPS-BENCHMARK CODE AS-IS, IT NEEDS TO BE ADJUSTED TO MATCH MY CODE'S CONVENTIONS, PARAMETERS AND FUNCTIONALITY
    @torch.no_grad()
    def conditioning_term(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        x_s, x_t, y = x_t, x_prev, measurement

        alpha_s = kwargs["alpha"]
        alpha_t = kwargs["alpha_prev"]
        scale = kwargs["scale"]

        def A(x):
            return self.operator.forward(x, **kwargs)

        with torch.enable_grad():
            y_hat = A(x_0_hat)

        def At(y):
            return torch.autograd.grad(y_hat, x_0_hat, y, retain_graph=True)[0]

        var_t = (1 - alpha_t) / torch.sqrt(alpha_t)
        var_y = self.noiser.sigma**2

        def cov_x_xt(v):
            return var_t * torch.autograd.grad(x_0_hat, x_t, v, retain_graph=True)[0]

        def cov_y_xt(v):
            return var_y * v + A(cov_x_xt(At(v)))

        error = y - y_hat
        grad = gmres(
            A=cov_y_xt,
            b=error,
            maxiter=self.maxiter,
        )
        grad = At(grad)
        grad = torch.autograd.grad(x_0_hat, x_t, grad)[0]
        grad = var_t * grad

        x_s = x_s + scale * grad

        return x_s, torch.linalg.vector_norm(error)

