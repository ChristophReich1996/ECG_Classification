from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from wettbewerb import load_references

if __name__ == '__main__':
    ecg_leads = load_references("../data/training/")[0]
    for ecg_lead_ in ecg_leads:
        if ecg_lead_.shape[0] == 18000:
            ecg_lead = torch.from_numpy(ecg_lead_).float()
            break
    print(ecg_lead.shape)
    s = torchaudio.transforms.Spectrogram(n_fft=64, win_length=64, hop_length=32, power=2, normalized=False)(ecg_lead)
    s = torch.log(s.clamp(min=1e-08))
    print(s.shape)
    plt.imshow(s, aspect="auto")
    plt.show()
    print(s.shape)
