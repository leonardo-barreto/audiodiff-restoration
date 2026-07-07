import torch
import torchaudio
import numpy as np

from scipy.signal import firls
from scipy.io import savemat
from typing import Optional, Tuple

def load_audio(file: str, rate: float):
        audio, fs = torchaudio.load(file)
        if fs != rate:
            raise ValueError(f"Sampling rate {fs} of the file is not {rate}")
        audio = torch.mean(audio, dim=0).unsqueeze(0)
        return audio

def setup_filters(
    filter_args: dict, fs: float
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    freqs = get_filter_freqs(fs, filter_args["points_per_band"])
    gains = get_filter_gains(
        filter_args["points_per_band"], filter_args["bandpass_power"]
    )
    inverse_gains = 1 / gains
    prefilter, postfilter = None, None
    if filter_args["use_prefilter"]:
        prefilter = firls(filter_args["order"], freqs, gains, fs=fs)
        prefilter = torch.Tensor(prefilter)
    if filter_args["use_postfilter"]:
        postfilter = firls(filter_args["order"], freqs, inverse_gains, fs=fs)
        postfilter = torch.Tensor(postfilter)
    if filter_args["save_path"] is not None:
        matfile = {
            "prefilter": prefilter if prefilter is not None else [],
            "postfilter": postfilter if postfilter is not None else [],
        }
        savemat(filter_args["save_path"], matfile)
    return prefilter, postfilter

def get_filter_freqs(fs: float, points_per_band: int):
    reject_end = (fs / 4) * 0.9
    pass_start = (fs / 4) * 1.1
    reject_band = np.linspace(0, reject_end, num=points_per_band)
    transition_band = np.linspace(reject_end, pass_start, num=points_per_band)
    pass_band = np.linspace(pass_start, fs / 2, num=points_per_band)
    return np.concatenate((reject_band, transition_band, pass_band))


def get_filter_gains(points_per_band: int, power_gain: float):
    reject_band = np.ones(points_per_band)
    pass_band = (power_gain ** (1 / 2)) * np.ones(points_per_band)
    transition_band = np.sqrt(np.linspace(1, power_gain, points_per_band))
    return np.concatenate((reject_band, transition_band, pass_band))

def filter_signal(signal: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
    denominator = torch.zeros(fir.shape)
    denominator[0] = 1
    return torchaudio.functional.filtfilt(signal, denominator, fir, clamp=False)