import torch
import os

if __name__ == '__main__':
    for folder in os.listdir(path="../experiments"):
        print(folder)
        try:
            acc = torch.load(os.path.join("../experiments", folder, "metrics", "Accuracy.pt"))
            f1 = torch.load(os.path.join("../experiments", folder, "metrics", "F1.pt"))
            print("ACC:", acc.max().item())
            print("F1:", f1[acc.argmax()].item())
        except:
            pass