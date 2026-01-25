import torch
from torch.utils.data import DataLoader
from datasets.maestro_dataset import MaestroDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------- user-adjustable parameters --------
    maestro_root = "/home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/datasets/maestro-v3.0.0"   # folder containing MAESTRO wavs
    fs = 44100
    seg_len = 2*65536
    seg_per_file = 8

    batch_size = 2
    num_workers = 4
    num_batches_to_check = 8
    # --------------------------------------------

    dataset = MaestroDataset(
        fs=fs,
        path=maestro_root,
        load_len=seg_len,
        seg_per_file=seg_per_file,
        seed=1
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Starting sanity check...\n")

    for i, (audio, sr, filename) in enumerate(loader):
        print(f"Batch {i}")
        print(f"  file        : {filename}")
        print(f"  audio shape : {audio.shape}")
        print(f"  audio dtype: {audio.dtype}")
        print(f"  sr         : {sr}")

        # --- assertions / checks ---
        assert audio.ndim == 2, "Expected (B, T)"
        assert audio.shape[0] == batch_size, "Unexpected batch size"
        assert audio.shape[1] == seg_len, "Segment length mismatch"
        assert torch.all(sr == fs), "Sampling rate mismatch"
        assert not torch.isnan(audio).any(), "NaNs found in audio"

        if i + 1 >= num_batches_to_check:
            break

    print("\nSanity check passed.")

if __name__ == "__main__":
    main()
