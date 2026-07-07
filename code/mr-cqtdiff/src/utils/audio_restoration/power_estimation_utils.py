import torch
import math
import torchaudio
from torch.nn.functional import conv1d

#From BVM-Restoration
XI_REGRESSION_INTERCEPT = -4.0884532164
XI_REGRESSION_SLOPE = -1.0245766357

def get_floor_and_ceil_powers(args: dict):
        print("\nGetting maximum and minimal signal powers ...")
        signal, _ = torchaudio.load(args.noisy_audio)
        powers = get_powers(signal, args.inference.denoising.power_window_size)
        print("\n")
        return torch.min(powers).item(), torch.max(powers).item()


def get_snr_range_min(block: torch.Tensor, floor: float, window_size: int):
    block_powers = get_powers(block, window_size)
    ratio = (torch.max(block_powers).item() - floor) / (max(floor, 1e-15))
    return 10 * math.log10(ratio)

def get_snr_range_width(floor: float, ceil: float):
    reduction = 10 * math.log10(max(floor, 1e-12) / ceil)
    return 43 + max(-40, reduction)

def get_powers(signal: torch.Tensor, window_size: int):
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.view(1, 1, -1).to(signal.device)
    squared_signal = (torch.pow(signal, 2)).view(1, signal.shape[0], -1)
    powers = conv1d(squared_signal, kernel)
    return powers.squeeze()

def get_xi_estimate(floor: float):
    return max(0.35, XI_REGRESSION_SLOPE * math.log10(floor) + XI_REGRESSION_INTERCEPT)
    