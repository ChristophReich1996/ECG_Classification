import numpy as np
from ecg_classification import ECGDataset
from wettbewerb import load_references

if __name__ == '__main__':
    # Init dataset
    ecg_leads, ecg_labels, fs, ecg_names = load_references("../data/training2017/")
    dataset = ECGDataset(ecg_leads, ecg_labels)
    # Init distribution matrix
    distribution = np.zeros(4, dtype=np.float32)
    # Iterate over dataset
    for _, _, labels in dataset:
        # Make distribution
        distribution[labels.argmax().numpy()] += 1.
    print(distribution)
    print(distribution / distribution.sum())
    print(1. - (distribution / distribution.sum()))
