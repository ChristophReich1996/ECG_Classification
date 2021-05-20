from wettbewerb import load_references
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Get data
    ecg_leads_1, ecg_labels_1, _, _ = load_references(
        "../data/training2017/")
    ecg_leads_2, ecg_labels_2, _, _ = load_references(
        "../data/training/")
    
    # Init list to store matched indexes
    matched_indexes = []

    # Check for matches
    matches = 0
    for index_1, (ecg_lead_1, ecg_label_1) in enumerate(zip(ecg_leads_1, ecg_labels_1)):
        for index_2, ecg_lead_2, ecg_label_2 in enumerate(zip(ecg_leads_2, ecg_labels_2)):
            if ecg_lead_1.shape == ecg_lead_2.shape:
                if np.allclose(ecg_lead_1, ecg_lead_2):
                    print("Match found")
                    print(ecg_label_1, ecg_label_2)
                    matched_indexes.append(index_1)
                    del ecg_leads_2[index_2]
                    del ecg_labels_2[index_2]
                    matches += 1
                    break
    print(matches)
    print(matched_indexes)