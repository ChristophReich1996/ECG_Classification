import matplotlib.pyplot as plt
import tikzplotlib

from wettbewerb import load_references
from ecg_classification.augmentation import AugmentationPipeline



if __name__ == '__main__':
    # Get data
    ecg_leads, ecg_labels, fs, ecg_names = load_references("D:/ECG_Classification Data/training/")
    # Init augmentation pipeline
    # Plot a few data samples
    for ecg_lead, ecg_label in enumerate(zip(ecg_leads, ecg_labels)):


        ecg = AugmentationPipeline()
        ecg.scale(ecg_lead)
        # Limit x- and y-axis of plot
        plt.xlim(0, 1000)
        plt.ylim(-150, 300)

        # Label x- and y-axis of plot
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
        plt.plot(ecg)


        tikzplotlib.save("augmented_plot_scale.tex")
        pass