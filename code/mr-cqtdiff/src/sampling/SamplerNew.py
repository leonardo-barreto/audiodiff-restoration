from tqdm import tqdm
import torch
import hydra
import os
import random

import utils.audio_restoration.denoising_utils as denoising_utils

class SamplerNew():

    def __init__(self, args, model, diff_params, conditioner = None):

        self.model = model.eval() #is it ok to do this here?
        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args
        
        if self.args.tester.sampling_params.same_as_training:
            self.sde_hp = diff_params.sde_hp
        else:
            self.sde_hp = self.args.tester.sampling_params.sde_hp

        # diffusion timestep params
        self.T = self.args.tester.sampling_params.T
        self.step_counter = 0

        # stochasticity parameters
        self.Schurn = self.args.tester.sampling_params.Schurn
        self.Snoise = self.args.tester.sampling_params.Snoise
        self.Stmin = self.args.tester.sampling_params.Stmin
        self.Stmax = self.args.tester.sampling_params.Stmax

        # order of the sampler
        self.order = self.args.tester.sampling_params.order

        # Possibility to set conditioner from outside, otherwise instantiate them from the tester config
        if conditioner is None:
            self.conditioner = hydra.utils.instantiate(self.args.conditioner, self.args, self.model, self.diff_params)
        else:
            self.conditioner = conditioner

    ##########################################
    # UTILITIES
    ##########################################
    
    def set_device(self, device):
        self.device = device

    ##########################################
    # NOISE SCHEDULE
    ##########################################

    def create_schedule(self, sigma_min=None, sigma_max=None, rho=None, T=None):
        """
        EDM schedule by default
        """
        if T is None:
            T=self.T

        if self.args.tester.sampling_params.schedule == "edm":
            if sigma_min is None:
                sigma_min = self.sde_hp.sigma_min
            if sigma_max is None:
                sigma_max = self.sde_hp.sigma_max
            if rho is None:
                rho = self.sde_hp.rho
            a = torch.arange(0, T)
            t = (sigma_max**(1/rho) + a/(T-1) *(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
            t[-1] = 0
            return t
        
        elif self.args.tester.sampling_params.schedule == "song":
            if sigma_min is None:
                sigma_min = self.sde_hp.sigma_min
            if sigma_max is None:
                sigma_max = self.sde_hp.sigma_max
            if rho is None:
                rho = self.sde_hp.rho
            eps = 0. if not "t_eps" in self.args.tester.diff_params.keys() else self.args.tester.diff_params.t_eps
            a = torch.arange(eps, T+1)
            t = sigma_min**2 * (sigma_max / sigma_min)**(2*a)
            t[-1] = 0
            return t    
        elif self.args.tester.sampling_params.schedule == "FM":
            t = torch.linspace(1, 0, T+1)
            return t

        else:
            raise NotImplementedError(f"schedule {self.args.tester.sampling_params.schedule} not implemented")

    def get_gamma(self, t):
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N = t.shape[0]
        gamma = torch.zeros(t.shape).to(t.device)

        # If desired, only apply stochasticity between a certain range of noises Stmin is 0 by default and Stmax is a huge number by default. (Unless these parameters are specified, this does nothing)
        indexes = torch.logical_and(t > self.Stmin, t < self.Stmax)

        # We use Schurn=5 as the default in our experiments
        gamma[indexes] = gamma[indexes] + torch.min(torch.Tensor([self.Schurn / N, 2 ** (1 / 2) - 1]))

        return gamma
    
    ##########################################
    # SCORES
    ##########################################

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

    # Added by Leonardo Barreto
    def get_score(self, x, t_i, y=None, degradation=None):
        """Compute the score for the current latent.

        Unconditional score is the gradient of log p(x).
        Conditional score is log p(x|y) = log p(x) + log p(y|x) and is implemented
        by adding a conditioning term from the configured conditioner.
        """

        with torch.no_grad():
            x_den = self.get_Tweedie_estimate(x, t_i)
            score = self.Tweedie2score(x_den, x, t_i)

        if self.conditioner is None:
            return score
        else:
            requires_grad = getattr(self.conditioner, "requires_grad")

            if requires_grad:
                with torch.enable_grad():
                    # copy of x with gradients enabled, so we can compute the conditioning term
                    x_grad = x.detach().requires_grad_(True)
                    x_den_grad = self.get_Tweedie_estimate(x_grad, t_i)
                    cond_score = self.conditioner.conditioning_term(x_grad, x_den_grad, t_i, y, degradation)
            else:
                cond_score = self.conditioner.conditioning_term(x, x_den, t_i, y, degradation)

            if not torch.is_tensor(cond_score):
                cond_score = torch.tensor(cond_score, dtype=score.dtype, device=score.device)

            # conditional score: ∇x log p(x|y)
            return score + cond_score

    ##########################################
    # DIFFUSION STEP
    ##########################################

    def stochastic_timestep(self, x, t, gamma, Snoise=1):
        t_hat = t + gamma * t  # if gamma_sig[i]==0 this is a deterministic step, make sure it doed not crash
        t_hat=torch.clamp(t_hat, 0, self.diff_params.max_t)
        epsilon = torch.randn(x.shape).to(x.device) * Snoise  # sample Gaussiannoise, Snoise is 1 by default
        if t_hat <= t:
            x_hat = x
        else:
            x_hat = x + ((t_hat ** 2 - t ** 2) ** (1 / 2)) * epsilon  # Perturb data
        return x_hat, t_hat

    def step(self, x_i, t_i, t_iplus1, gamma_i, y=None, degradation=None):

        with torch.no_grad():
            x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i, self.Snoise)
            score = self.get_score(x_hat, t_hat, y, degradation)
            ode_integrand = self.diff_params._ode_integrand(x_hat, t_hat, score)
            dt = t_iplus1 - t_hat

            if t_iplus1 != 0 and self.order == 2:  # second order correction
                t_prime = t_iplus1
                x_prime = x_hat + dt * ode_integrand
                score = self.get_score(x_prime, t_prime, y, degradation)
                ode_integrand_next = self.diff_params._ode_integrand(x_prime, t_prime, score)
                ode_integrand_midpoint = .5 * (ode_integrand + ode_integrand_next)
                x_iplus1 = x_hat + dt * ode_integrand_midpoint
            else:
                x_iplus1 = x_hat + dt * ode_integrand

            return x_iplus1
    
    ##########################################
    # MAIN PREDICTION LOOP
    ##########################################

    def predict_unconditional(
        self,
        shape,  # observations (lowpssed signal) Tensor with shape ??
        device
    ):

        print(shape)
        return self.predict(shape, device, y=None, degradation=None)

    def predict_conditional(
        self,
        y,  # observations (lowpssed signal) Tensor with shape ??
        degradation  # lambda function
    ):
        return self.predict(y.shape, y.device, y=y, degradation=degradation)

    def predict_resample(
        self,
        y,
        shape,
        degradation,
    ):
        return self.predict(shape, y.device, y=y, degradation=degradation)

    def predict(
            self,
            shape,  # observations (lowpssed signal) Tensor with shape ??
            device,
            y=None,
            degradation=None
    ):

        # get the noise schedule
        t = self.create_schedule().to(device).to(torch.float32)

        # sample prior
        x = self.diff_params.sample_prior(t=t[0], shape=shape).to(device).to(torch.float32)

        x_init = x.clone()

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma = self.get_gamma(t).to(device)

        for i in tqdm(range(0, self.T-1, 1)):
            self.step_counter = i
            x = self.step(x, t[i], t[i + 1], gamma[i], y=y, degradation=degradation)

        return x.detach(), x_init.detach()

class Sampler78rpm(SamplerNew):
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
            #res = self.predict_conditional(y_noisy, degradation)
            #return res
            return self.predict_conditional(y_noisy, degradation)

    def predict_gramophone_denoising_with_snr(self, y_noisy, range):
        degradation = lambda x: self.add_noise_with_snr(x, range)
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            #res = self.predict_conditional(y_noisy, degradation)
            #return res
            return self.predict_conditional(y_noisy, degradation)

    # This is gonna need conditioner integration
    def predict_gramophone_denoising_with_variable_xi(self, y_noisy, range):
        degradation = lambda x: self.add_noise_with_variable_xi(x, range)
        if self.rid:
            res, denoised, t = self.predict_conditional(y_noisy, degradation)
            return res, denoised, t
        else:
            #res = self.predict_conditional(y_noisy, degradation)
            #return res
            return self.predict_conditional(y_noisy, degradation)
    