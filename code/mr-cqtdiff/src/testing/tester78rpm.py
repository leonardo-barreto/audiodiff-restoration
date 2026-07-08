import hydra
import torch
import copy
import math
from tqdm import tqdm

from datasets.inference_audio_set import SingleAudioDataset
from torch.utils.data import DataLoader

from typing import Optional, Tuple

import utils.audio_restoration.audio_utils as audio_utils
import utils.audio_restoration.power_estimation_utils as power_estimation_utils
import utils.audio_restoration.block_restoration as block_restoration

from testing.tester import Tester

class Tester78rpm(Tester): # Inheritance just to use the base methods, such as load_checkpoint and etc.
    def __init__(
            self, args, network, diff_params, sampler=None, noisy_audio_blocks: Optional[DataLoader] = None, device=None):

        # Parent class-only attributes (don't care)
        self.in_training = False
        self.use_wandb = False
        self.inference_train_set = None
        self.inference_test_set = None

        # Relevant attributes for this class
        self.args = args
        self.network = network
        self.diff_params = copy.copy(diff_params)
        torch.backends.cudnn.benchmark = True
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Setup the noisy audio blocks and output length
        self.noisy_audio_blocks = noisy_audio_blocks
        self.out_len: Optional[int] = None

        # Setup the denoised audio blocks tensor and final reconstructed audio
        self.denoised_audio_blocks = torch.Tensor().to("cpu")
        self.reconstructed_audio = torch.Tensor().to("cpu")

        # Possibility to set sampler from outside, otherwise instantiate them from the tester config
        if sampler is None:
            self.sampler = hydra.utils.instantiate(args.tester.sampler, self.args, self.network, self.diff_params)
    
    def inference(self):

        window_size = self.args.inference.denoising.power_window_size
        floor, ceil = power_estimation_utils.get_floor_and_ceil_powers(self.args)

        # THIS NEEDS TO CHANGE TO THE CONDITIONER
        #if self.args.inference.infer_xi == True:
            #sampler.xi = power_estimation_utils.get_xi_estimate(floor)
            #print(f"XI Estimate: {sampler.xi}")
        
        # Loop through the blocks and denoise them
        denoised_blocks = []
        for block in tqdm(self.noisy_audio_blocks):
            b = block[0].to(self.device)

            # Perform inference with selected noise injection method
            if (self.args.inference.use_range == True) and (self.args.inference.use_gain == True):
                print(f"Denoising with fixed SNR Range {self.args.inference.snr_range}")
                denoised_block, noise_init = self.sampler.predict_gramophone_denoising_with_snr(
                    b, self.args.inference.snr_range
                )
            if (self.args.inference.use_range == False) and (self.args.inference.use_gain == False):
                print(f"Denoising with fixed gain {self.args.inference.noise_gain}")
                denoised_block, noise_init = self.sampler.predict_gramophone_denoising(
                    b, self.args.inference.noise_gain
                )
            if (self.args.inference.use_range == True) and (self.args.inference.use_gain == False):
                print("Denoising estimating SNR range")
                range_min = power_estimation_utils.get_snr_range_min(b, floor, window_size)
                range_width = power_estimation_utils.get_snr_range_width(floor, ceil)
                snr_range = [range_min, range_min + range_width]
                print(f"Estimated SNR Range: [{snr_range[0]},{snr_range[1]}]")
                denoised_block, noise_init = self.sampler.predict_gramophone_denoising_with_snr(b, snr_range)
            if (self.args.inference.use_range == False) and (self.args.inference.use_gain == True):
                print("Denoising estimating noise gain")
                print(f"Estimated gain: {math.sqrt(floor)}\n")
                denoised_block, noise_init = self.sampler.predict_gramophone_denoising(b, math.sqrt(floor))

            #denoised_block = b # bypass for block testing

            # unsqueeze the denoised block to add a batch dimension and concatenate it to the denoised_blocks tensor    
            denoised_block = denoised_block.unsqueeze(0)
            denoised_blocks.append(denoised_block)

        # Bring the denoised blocks back to CPU and reconstruct the audio
        self.denoised_audio_blocks = torch.cat(denoised_blocks, dim=0).cpu()
        self.reconstructed_audio = self.reconstruct(self.denoised_audio_blocks, self.out_len)

        return self.reconstructed_audio
    
    def reconstruct(self, blocks: torch.Tensor, audio_len: int) -> torch.Tensor:
        block_size = int(self.args.block_duration * self.args.sample_rate)
        superposition = int(self.args.overlap_duration * self.args.sample_rate)
        window = block_restoration.windows.window_with_fade(block_size, superposition)
        reconstructed = block_restoration.synthesis.overlap_and_add(blocks, window, superposition)
        pad_size = reconstructed.shape[-1] - audio_len
        left_pad_size = pad_size // 2
        right_pad_size = pad_size // 2 + pad_size % 2
        reconstructed = reconstructed[:, left_pad_size:-(right_pad_size)]
        return reconstructed
    
    def setup_data(self, prefilter: Optional[torch.Tensor] = None):
        audio = audio_utils.load_audio(self.args.noisy_audio, self.args.sample_rate)
        if prefilter is not None:
            audio = audio_utils.filter_signal(audio, prefilter)
        data = SingleAudioDataset(audio, self.args)
        self.noisy_audio_blocks = DataLoader(data, batch_size=1, shuffle=False)
        self.out_len = audio.shape[-1]

        return self.noisy_audio_blocks, self.out_len
    
    


    
