import os
import random

import torch

import utils.audio_restoration.denoising_utils as denoising_utils

from sampling.BaseSampler import BaseSampler


class GramophoneDegradationModel:
    """Fixed affine gramophone degradation for one diffusion trajectory."""

    def __init__(
        self,
        noises,
        num_steps,
        gain=None,
        snr_range=None,
        update_xi=False,
        conditioner=None,
    ):
        if not noises:
            raise ValueError("At least one gramophone noise sample is required")
        if gain is None and snr_range is None:
            raise ValueError("Either gain or snr_range must be provided")
        if gain is not None and snr_range is not None:
            raise ValueError("gain and snr_range are mutually exclusive")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if update_xi and conditioner is None:
            raise ValueError("conditioner is required when update_xi is enabled")

        # Gets noise samples and samples a vector of T indices to select a noise sample for each diffusion step
        self.noises = noises
        self.noise_indices = [
            random.randrange(len(noises)) for _ in range(num_steps)
        ]

        # Sets other parameters
        self.gain = gain
        self.snr_range = snr_range
        self.update_xi = update_xi
        self.conditioner = conditioner
        self.noise = None
        self.snr = None

    def setup_degradation(self, step, x_i=None, t_i=None):
        if step >= len(self.noise_indices):
            raise IndexError(f"Diffusion step {step} is outside the degradation schedule")

        # Select noise sample and normalize it to unit power
        self.noise = self.noises[self.noise_indices[step]].to(
            x_i.device if x_i is not None else self.noises[0].device
        )
        noise_power = torch.mean(self.noise.square())
        if noise_power <= 0:
            raise ValueError("Selected gramophone noise has zero power")
        self.noise = self.noise / torch.sqrt(noise_power)

        # If on SNR mode, sample a new SNR value and update xi if required
        if self.snr_range is not None:
            self.snr = random.uniform(*self.snr_range)
            if self.update_xi:
                self.conditioner.xi = max(0.35, 0.105 * self.snr - 0.7)

    def _require_step(self):
        if self.noise is None:
            raise RuntimeError(
                "setup_degradation must be called before evaluating degradation"
            )

    def forward(self, x):
        self._require_step()
        if self.snr_range is not None:
            return denoising_utils.add_noise_with_snr(x, self.noise, self.snr)
        else:
            return denoising_utils.add_noise_with_gain(x, self.noise, self.gain)

    def __call__(self, x):
        return self.forward(x)

    def jvp(self, x, vector):
        self._require_step()
        return denoising_utils.gramophone_degradation_jvp(x, vector)

    def vjp(self, x, vector):
        self._require_step()
        return denoising_utils.gramophone_degradation_vjp(x, vector)


class Sampler78rpm(BaseSampler):
    def __init__(
        self, args, model, diff_params, conditioner=None, rid=False
    ):
        super().__init__(args, model, diff_params, conditioner)
        self.rid = rid
        self.get_noise_samples()

    ##############################
    # GRAMOPHONE NOISE RETRIEVAL #
    ##############################

    def get_noise_samples(self):
        # samples need to be in the same sampling rate as original audio !!!
        noise_folder = self.args.inference.denoising.noise_files_path
        noise_paths = [
            os.path.join(noise_folder, file) for file in os.listdir(noise_folder)
        ]
        self.noises = [
            denoising_utils.get_noise(
                n_samples=self.args.audio_len,
                crossfade_time=self.args.inference.denoising.crossfade_duration,
                noise_path=path,
            )
            for path in noise_paths
        ]

    ##############################
    # GRAMOPHONE NOISE INJECTION #
    ##############################

    ########################
    # PREDICTION FUNCTIONS #
    ########################

    def predict_gramophone_denoising(self, y_noisy, gain):
        degradation = GramophoneDegradationModel(
            noises=self.noises,
            num_steps=self.T,
            gain=gain,
            conditioner=self.conditioner,
        )
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            return self.predict_conditional(y_noisy, degradation)

    def predict_gramophone_denoising_with_snr(self, y_noisy, range):
        degradation = GramophoneDegradationModel(
            noises=self.noises,
            num_steps=self.T,
            snr_range=range,
            conditioner=self.conditioner,
        )
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            return self.predict_conditional(y_noisy, degradation)

    # This is gonna need conditioner integration
    def predict_gramophone_denoising_with_variable_xi(self, y_noisy, range):
        degradation = GramophoneDegradationModel(
            noises=self.noises,
            num_steps=self.T,
            snr_range=range,
            update_xi=True,
            conditioner=self.conditioner,
        )
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            return self.predict_conditional(y_noisy, degradation)
