import torch
from typing import Optional

import utils.audio_restoration.power_estimation_utils as power_estimation_utils
from conditioners.BaseConditioner import BaseConditioner

class DPS(BaseConditioner):

    def __init__(self, args, model, diff_params, infer_xi=None, xi=None, max_thresh_grads=None, requires_grad=True):
        super().__init__(args, model, diff_params, requires_grad)

        #self.infer_xi = self.args.conditioner.infer_xi if infer_xi is None else infer_xi
        self.infer_xi = infer_xi
        #self.threshold_on_grads = self.args.conditioner.max_thresh_grads if max_thresh_grads is None else max_thresh_grads
        self.threshold_on_grads = max_thresh_grads
        print(f"Conditioner: DPS with infer_xi={self.infer_xi} and threshold_on_grads={self.threshold_on_grads}")

        if self.infer_xi == True:
            floor, ceil = power_estimation_utils.get_floor_and_ceil_powers(self.args)
            self.xi = power_estimation_utils.get_xi_estimate(floor)
            print(f"XI Estimate: {self.xi}")
        else:
            #self.xi = self.args.conditioner.xi if xi is None else xi
            self.xi = xi
            print(f"XI Set: {self.xi}")
    
    def conditioning_term(self, x, x_den, t_i, y, degradation):
        #x.requires_grad_()
        #x_hat = self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
        
        den_rec = degradation(x_den)

        if len(y.shape) == 3:
            dim = (1, 2)
        elif len(y.shape) == 2:
            dim = 1

        norm = torch.linalg.norm(y - den_rec, dim=dim, ord=2)

        rec_grads = torch.autograd.grad(outputs=norm, inputs=x)
        rec_grads = rec_grads[0]

        normguide = torch.linalg.norm(rec_grads) / self.args.audio_len**0.5

        # normalize scaling
        s = self.xi / (normguide * t_i + 1e-6)

        # optionally apply a threshold to the gradients
        if self.threshold_on_grads > 0:
            # apply thresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts
            rec_grads = torch.clip(
                rec_grads, min=-self.threshold_on_grads, max=self.threshold_on_grads
            )

        # conditioning term: scaled guidance
        return (-s*rec_grads)

