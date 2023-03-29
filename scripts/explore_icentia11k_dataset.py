import gzip
import pickle
import os

if __name__ == '__main__':
    # Load sample
    with gzip.open("/mnt/scratch/creich/icentia11k/00000_batched.pkl.gz", "rb") as file:
        data = pickle.load(file)
    print(data.shape)

    # Load label
    with gzip.open("/mnt/scratch/creich/icentia11k/00200_batched_lbls.pkl.gz", "rb") as file:
        lbls = pickle.load(file)
    print(lbls)


    classes = []
    noise = 0
    nsr = 0
    afib = 0
    afl = 0
    for index, file in enumerate(os.listdir("/mnt/scratch/creich/icentia11k")):
        if "lbls" in file:
            with gzip.open(os.path.join("/mnt/scratch/creich/icentia11k", file), "rb") as file:
                lbls = pickle.load(file)
                for sample in lbls:
                    rtype = sample["rtype"]
                    noise += (len(rtype[2]) + len(rtype[1]) + len(rtype[0]))
                    nsr += len(rtype[3])
                    afib += len(rtype[4])
                    afl += len(rtype[5])
        print(index)
    print(noise, nsr, afib, afl)