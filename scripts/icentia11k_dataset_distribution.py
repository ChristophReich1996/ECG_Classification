import numpy as np
from torch.utils.data import DataLoader
from ecg_classification import Icentia11kDataset, icentia11k_dataset_collate_fn

if __name__ == '__main__':
    # Init dataset
    dataset = DataLoader(Icentia11kDataset(path="/home/creich/scratch/icentia11k", split=list(range(11000))),
                         num_workers=12, collate_fn=icentia11k_dataset_collate_fn, batch_size=1)
    # Init distribution matrix
    distribution = np.zeros(7, dtype=np.float32)
    # Iterate over dataset
    for index, (ecg_lead, spectrogram, labels) in enumerate(dataset):
        print(index, ecg_lead.shape, spectrogram.shape)
        # Make distribution
        for label in labels:
            distribution[label.argmax().numpy()] += 1.
    print(distribution)
    print(distribution / distribution.sum())
    print(1. - (distribution / distribution.sum()))
