import torch
from torch.utils.data import Dataset
import utils.audio_restoration.block_restoration as block_restoration


class SingleAudioDataset(Dataset):
    def __init__(self, audio: torch.Tensor, args: dict):
        self.audio = audio
        self.block_duration = args.block_duration
        self.overlap_duration = args.overlap_duration
        self.fs = args.sample_rate
        self._divide_signal()

    def _divide_signal(self):
        superposition = int(self.overlap_duration * self.fs)
        block_size = int(self.block_duration * self.fs)
        window = block_restoration.windows.flat_window(block_size)
        self.blocks = block_restoration.analysis.divide_signal(self.audio, window, superposition)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]
