import torch

from utils.linalg import gmres
import utils.audio_restoration.power_estimation_utils as power_estimation_utils
from .BaseConditioner import BaseConditioner


class MomentMatching(BaseConditioner):
    """Moment Matching conditioning for the EDM score parameterization."""

    def __init__(
        self,
        args,
        model,
        diff_params,
        maxiter=1,
        infer_xi=None,
        xi=None,
        max_thresh_grads=None,
        measurement_noise_sigma=0.0,
        requires_grad=True,
    ):
        super().__init__(args, model, diff_params, requires_grad)

        self.infer_xi = infer_xi
        self.threshold_on_grads = max_thresh_grads
        self.maxiter = maxiter
        self.measurement_noise_sigma = measurement_noise_sigma

        if self.maxiter < 1:
            raise ValueError("maxiter must be positive")
        if self.measurement_noise_sigma < 0:
            raise ValueError("measurement_noise_sigma must be non-negative")

        if self.infer_xi:
            floor, _ = power_estimation_utils.get_floor_and_ceil_powers(self.args)
            self.xi = power_estimation_utils.get_xi_estimate(floor)
            print(f"XI Estimate: {self.xi}")
        else:
            self.xi = xi
            print(f"XI Set: {self.xi}")

        print(
            "Conditioner: MomentMatching "
            f"with maxiter={self.maxiter}, xi={self.xi}, "
            f"measurement_noise_sigma={self.measurement_noise_sigma}"
        )

    @torch.no_grad()
    def conditioning_term(self, x, x_den, t_i, y, degradation):
        # Validate the degradation interface required by the measurement-space solve.
        if not all(hasattr(degradation, name) for name in ("forward", "jvp", "vjp")):
            raise TypeError("MomentMatching requires a degradation with forward, jvp, and vjp")

        with torch.enable_grad():
            # Evaluate the fixed-step affine degradation and form the measurement residual.
            y_hat = degradation.forward(x_den)
            error = y - y_hat

            # In EDM, x_t = x_0 + sigma_t * epsilon, so the covariance coefficient
            # corresponding to the benchmark's var_t is sigma_t^2.
            diffusion_variance = t_i.square()

            def cov_x_xt(vector):
                # Propagate a measurement-space vector through Cov(x_0, x_t).
                with torch.enable_grad():
                    return diffusion_variance * torch.autograd.grad(
                        x_den,
                        x,
                        vector,
                        retain_graph=True,
                    )[0]

            def cov_y_xt(vector):
                # Apply the measurement covariance operator used by GMRES.
                with torch.enable_grad():
                    adjoint = degradation.vjp(x_den, vector)
                    propagated = cov_x_xt(adjoint)
                    return (
                        self.measurement_noise_sigma**2 * vector
                        + degradation.jvp(x_den, propagated)
                    )

            # Avoid solving a zero-residual system and return no conditioning signal.
            if torch.count_nonzero(error) == 0:
                return torch.zeros_like(x)

            # Solve Cov(y, x_t) * v = y - y_hat in measurement space.
            measurement_gradient = gmres(
                A=cov_y_xt,
                b=error,
                maxiter=self.maxiter,
            )

            # Pull the solved measurement direction back through the degradation
            # and the Tweedie estimate to obtain a latent-space gradient.
            latent_gradient = degradation.vjp(x_den, measurement_gradient)
            gradient = torch.autograd.grad(
                x_den,
                x,
                latent_gradient,
            )[0]
            # Not needed if returning a score correction instead of updating the diffusion state.
            #gradient = diffusion_variance * gradient

        # Apply the same xi-based normalization used by DPS.
        normguide = torch.linalg.vector_norm(gradient) / self.args.audio_len**0.5
        scale = self.xi / (normguide * t_i + 1e-6)

        #quick manual override
        #scale = self.xi

        # Retain DPS's gradient clipping heuristic before scaling the correction.
        if self.threshold_on_grads is not None and self.threshold_on_grads > 0:
            gradient = torch.clip(
                gradient,
                min=-self.threshold_on_grads,
                max=self.threshold_on_grads,
            )

        # Return a score correction; unlike MMPS, do not update the diffusion state.
        return scale * gradient
