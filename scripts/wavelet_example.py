import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from wettbewerb import load_references

if __name__ == '__main__':
    ecg_leads = load_references("../data/training/")[0]
    s, a = pywt.cwt(ecg_leads[0].astype(float), range(1, 501), wavelet="gaus1")
    print(s.shape, ecg_leads[0].shape, a.shape)
    plt.imshow(s[:, :1000], aspect="auto")
    plt.colorbar()
    plt.show()
