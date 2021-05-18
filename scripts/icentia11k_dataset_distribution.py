import numpy as np
from ecg_classification import Icentia11kDataset

if __name__ == '__main__':
    # Init dataset
    dataset = Icentia11kDataset(path="E:\\ECG_Data\\icentia11k", split=list(range(11000)))
    # Init distribution matrix
    distribution = np.zeros(7, dtype=np.float32)
    # Iterate over dataset
    for index, (_, _, labels) in enumerate(dataset):
        print(index)
        # Make distribution
        for label in labels:
            distribution[label.argmax().numpy()] += 1.
        if index == 100:
            break
    print(distribution)
    print(distribution / distribution.sum())
    print(1. - (distribution / distribution.sum()))
