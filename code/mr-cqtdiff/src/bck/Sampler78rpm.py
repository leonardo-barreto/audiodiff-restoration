import os
import random

import torch

import utils.audio_restoration.denoising_utils as denoising_utils

from sampling.BaseSampler import BaseSampler


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

    def add_noise(self, signal, gain):
        noise = random.choice(self.noises).to(self.device)
        noise_power = torch.sum(torch.pow(noise, 2)) / noise.shape[-1]
        noise = noise / torch.sqrt(noise_power)
        noisy_signal = denoising_utils.add_noise_with_gain(signal, noise, gain)
        return noisy_signal

    def add_noise_with_snr(self, signal, range):
        noise = random.choice(self.noises).to(self.device)
        snr = random.uniform(range[0], range[1])
        noisy_signal = denoising_utils.add_noise_with_snr(signal, noise, snr)
        return noisy_signal

    # This is gonna need conditioner integration
    def add_noise_with_variable_xi(self, signal, range):
        noise = random.choice(self.noises).to(self.device)
        snr = random.uniform(range[0], range[1])
        self.xi = max(0.35, 0.105 * range[0] - 0.7)
        noisy_signal = denoising_utils.add_noise_with_snr(signal, noise, snr)
        return noisy_signal

    ########################
    # PREDICTION FUNCTIONS #
    ########################

    def predict_gramophone_denoising(self, y_noisy, gain):
        degradation = lambda x: self.add_noise(x, gain)
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            return self.predict_conditional(y_noisy, degradation)

    def predict_gramophone_denoising_with_snr(self, y_noisy, range):
        degradation = lambda x: self.add_noise_with_snr(x, range)
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            return self.predict_conditional(y_noisy, degradation)

    # This is gonna need conditioner integration
    def predict_gramophone_denoising_with_variable_xi(self, y_noisy, range):
        degradation = lambda x: self.add_noise_with_variable_xi(x, range)
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            return self.predict_conditional(y_noisy, degradation)
