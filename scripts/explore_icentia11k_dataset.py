import gzip
import pickle
import os

if __name__ == '__main__':
    # Load sample
    with gzip.open("E:\\ECG_Data\\icentia11k\\00000_batched.pkl.gz", "rb") as file:
        data = pickle.load(file)
    print(data.shape)

    # Load label
    with gzip.open("E:\\ECG_Data\\icentia11k\\00000_batched_lbls.pkl.gz", "rb") as file:
        lbls = pickle.load(file)
    print(lbls)

    classes = []
    for index, file in enumerate(os.listdir("E:\\ECG_Data\\icentia11k")):
        print(index)
        if "lbls" in file:
            with gzip.open(os.path.join("E:\\ECG_Data\\icentia11k", file), "rb") as file:
                lbls = pickle.load(file)
            for lbl in lbls:
                classes.append(sum([label.shape[0] != 0 for label in lbl["rtype"]]))
        if index > 1000:
            exit(22)