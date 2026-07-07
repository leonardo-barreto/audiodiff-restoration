import utils.audio_restoration.block_restoration as block_restoration
import torch

fs = 44100
signal_size = 65536
block_size = 44100
superposition = 11025

signal = torch.zeros([1,signal_size])

def pad_signal(
    signal: torch.Tensor, block_size: int, superposition: int
) -> torch.Tensor:
    superposition_pad = torch.zeros(signal.shape[0], superposition)
    print(f"superposition pad shape = {superposition_pad.shape}")
    signal = torch.cat((superposition_pad, signal, superposition_pad), dim=1)
    print(f"signal shape after superposition pad = {signal.shape}")
    pad_size = block_restoration.analysis.calculate_window_padding_size(
        signal.shape[-1], block_size, superposition
    )
    print(f"pad size = {pad_size}")
    left_padding = torch.zeros((signal.shape[0], pad_size // 2))
    right_padding = torch.zeros((signal.shape[0], pad_size // 2 + pad_size % 2))
    print(f"left padding shape = {left_padding.shape}")
    print(f"right padding shape = {right_padding.shape}")
    return torch.cat((left_padding, signal, right_padding), dim=1)

x = pad_signal(signal, block_size, superposition)
print(f"x shape = {x.shape}")