import torch

from wettbewerb import load_references

from ecg_classification import PhysioNetDataset, ECGCNN, ECGAttNet, ECGCNN_CONFIG_L, ECGAttNet_CONFIG_L

if __name__ == '__main__':

    ecg_leads, ecg_labels, fs, ecg_names = load_references("data/training/")
    dataset = PhysioNetDataset(ecg_leads=ecg_leads, ecg_labels=ecg_labels)
    network = ECGAttNet(config=ECGAttNet_CONFIG_L)

    print(sum([p.numel() for p in network.parameters()]))

    with torch.no_grad():
        for ecg_lead, spectrogram, label in dataset:
            output = network(ecg_lead=ecg_lead[None], spectrogram=spectrogram[None])
            print(output.shape)
            exit(22)

    exit(22)
