import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

from wettbewerb import load_references

if __name__ == '__main__':
    # Get data
    ecg_leads, ecg_labels, fs, ecg_names = load_references("../data/training/")
    # Init dict for plotting samples from all classes
    plot_dict = {"N": False, "O": False, "A": False, "~": False}
    # Set sample length
    sample_length = 1250
    # Plot a few data samples
    for index, (ecg_lead, ecg_label) in enumerate(zip(ecg_leads, ecg_labels)):
        if not plot_dict[ecg_label]:
            plot_dict[ecg_label] = True
            x = np.arange(0, sample_length, dtype=np.float32) / fs
            plt.plot(x, ecg_lead[-sample_length:])
            plt.xlabel("time/s")
            plt.ylabel("voltage/mV")
            tikzplotlib.save("{}_{}.tex".format(index, ecg_label), axis_width="\\figW", axis_height="\\figH")
            plt.show()