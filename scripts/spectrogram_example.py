from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from wettbewerb import load_references

if __name__ == '__main__':
    ecg_leads = load_references("../data/training/")[0]
    ecg_lead = torch.from_numpy(ecg_leads[0]).float()
    s = torchaudio.transforms.Spectrogram(n_fft=64, win_length=64, hop_length=32, power=2, normalized=True)(ecg_lead)
    s = torch.log(s.clamp(min=1e-08))
    print(s.shape)
    plt.imshow(s, aspect="auto")
    plt.show()
    print(s.shape)
