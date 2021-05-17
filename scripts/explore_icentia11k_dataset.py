import gzip
import pickle

if __name__ == '__main__':
    # Load sample
    with gzip.open("E:\\ECG_Data\\icentia11k\\00000_batched.pkl.gz", "rb") as file:
        data = pickle.load(file)
    print(data.shape)

    # Load label
    with gzip.open("E:\\ECG_Data\\icentia11k\\00000_batched_lbls.pkl.gz", "rb") as file:
        data = pickle.load(file)
    print(data)