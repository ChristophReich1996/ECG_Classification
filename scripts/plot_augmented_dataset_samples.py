import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
import torch

from wettbewerb import load_references
from ecg_classification import AugmentationPipeline, AUGMENTATION_PIPELINE_CONFIG, ECGDataset

if __name__ == '__main__':
    # Get data
    ecg_leads, ecg_labels, fs, ecg_names = load_references("../data/training/")
    # Get one sample of the dataset
    ecg_lead = torch.from_numpy(ecg_leads[1902]).float()
    print(ecg_labels[1902])
    # Normalize signal
    ecg_lead = (ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08)
    # Init augmentation pipeline
    augmentation_pipeline = AugmentationPipeline(config=AUGMENTATION_PIPELINE_CONFIG)
    # List augmentations
    augmentations = [augmentation_pipeline.scale, augmentation_pipeline.drop, augmentation_pipeline.cutout,
                     augmentation_pipeline.shift, augmentation_pipeline.resample, augmentation_pipeline.random_resample,
                     augmentation_pipeline.sine, augmentation_pipeline.band_pass_filter]
    # Set sample length
    sample_length = 2500
    # Iterate over augmentations
    for augmentation in augmentations:
        ecg_lead_augmented = augmentation(ecg_lead.clone()).numpy()
        x = np.arange(0, sample_length, dtype=np.float32) / fs
        plt.plot(x, ecg_lead_augmented[sample_length:2 * sample_length])
        plt.xlabel("time/s")
        plt.ylabel("voltage/mV")
        print(augmentation.__name__)
        tikzplotlib.save("{}.tex".format(augmentation.__name__), axis_width="\\figW", axis_height="\\figH")
        plt.show()
