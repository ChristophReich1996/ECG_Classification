import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch

from wettbewerb import load_references

if __name__ == '__main__':
    ecg_leads = load_references("../data/training/")[0]
    s, a = pywt.cwt(ecg_leads[0], range(1, 129), 'morl', 1)
    print(s.shape, ecg_leads[0].shape)
    plt.imshow(s[:, :127], aspect="auto")
    plt.colorbar()
    plt.show()
    print(torch.from_numpy(s).unfold(dimension=-1, size=128, step=128).shape)

    s, a = pywt.cwt(ecg_leads[0][:3000], range(1, 129), 'morl', 1)
    print(s.shape, ecg_leads[0][:3000].shape)
    plt.imshow(s[:, :127], aspect="auto")
    plt.colorbar()
    plt.show()
