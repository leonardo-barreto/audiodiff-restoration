import os
import random
import glob
import torch
import torchaudio
from torch.utils.data import IterableDataset

class MaestroDataset(IterableDataset):
    """
    MAESTRO IterableDataset matching the scope of toy_dataset.py.

    Responsibilities:
    - Scan folders
    - Randomly select a file
    - Make sure sampling rate matches the desired value (resampling should be done manually)
    - Randomly extract one audio segment of specified length

    """

    def __init__(self,
                 fs,
                 path,
                 load_len,
                 seg_per_file=1,
                 seed=42
                 ):
        super().__init__()
        self.seed = seed

        self.main_path = path
        #glob .wav files
        self.train_samples = glob.glob(os.path.join(self.main_path, '**', '*.wav'), recursive=True)
        if len(self.train_samples) == 0:
            raise RuntimeError(f"No WAV files found under {path}")

        self.seg_len=int(load_len)
        self.seg_per_file = seg_per_file
        self.fs = fs

    def load_audio(self, file_path):
        data, sr = torchaudio.load(file_path)
        # Mono conversion
        if data.shape[0] > 1:
            data = data.mean(dim=0)
        else:
            data = data.squeeze(0) # Matches shape if it's already mono
        if sr != self.fs:
            raise RuntimeError(f"File {os.path.basename(file_path)} is not at the desired {self.fs} Hz sampling rate")
        return data, sr

    def __iter__(self):
        # Worker sharding and seeding, runs once per worker, when iter() is created
        worker_info = torch.utils.data.get_worker_info()
        files = self.train_samples
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed
        rng = random.Random(worker_seed)
        # Data retrieval loop, ran by worker when requested for data
        while True:
            try:
                file_path = rng.choice(files)
                data, sr = self.load_audio(file_path)
                data_len = data.shape[-1]

                num_frames = data_len // self.seg_len
                if num_frames > (self.seg_per_file // 2):
                    for _ in range(self.seg_per_file):
                        # get seg_per_file random segments to be a bit faster
                        start = rng.randint(0, data_len - self.seg_len)
                        segment = data[..., start:start + self.seg_len]
                        yield segment, sr
                else:
                    # if audio file isn't long enough, just pass
                    pass
            except Exception as e:
                print("Error in a sample of MaestroDataset:", e)
                continue