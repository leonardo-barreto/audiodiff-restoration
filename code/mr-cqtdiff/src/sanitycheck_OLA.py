import sys
import torch
import torchaudio


def compare_wavs(file1, file2):
    wav1, sr1 = torchaudio.load(file1)
    wav2, sr2 = torchaudio.load(file2)

    assert sr1 == sr2, f"Different sample rates: {sr1} != {sr2}"
    assert wav1.shape == wav2.shape, (
        f"Different shapes: {wav1.shape} != {wav2.shape}"
    )

    if torch.equal(wav1, wav2):
        print("The sample arrays are exactly equal.")
        return

    rmse = torch.sqrt(torch.mean((wav1 - wav2) ** 2))
    print("The sample arrays are NOT equal.")
    print(f"RMSE: {rmse.item():.10f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_wavs.py file1.wav file2.wav")
        sys.exit(1)

    compare_wavs(sys.argv[1], sys.argv[2])