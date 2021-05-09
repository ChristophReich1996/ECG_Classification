from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pywt
import numpy as np


class ECGDataset(Dataset):
    """
    This class implements the ECG dataset for atrial fibrillation classification.
    """

    def __init__(self, ecg_leads: List[np.ndarray], ecg_labels: List[str],
                 augmentation_pipeline: Optional[nn.Module] = None, spectrogram_length: int = 18000,
                 normalize: bool = True, temporal_downscale: int = 8) -> None:
        """
        Constructor method
        :param ecg_leads: (List[np.ndarray]) ECG data as list of numpy arrays
        :param ecg_labels: (List[str]) ECG labels as list of strings (N, O, A, ~)
        :param augmentation_pipeline: (Optional[nn.Module]) Augmentation pipeline
        :param spectrogram_length: (int) Fixed spectrogram length (achieved by zero padding)
        :param normalize: (bool) If true signal is normalized to a mean and std of zero and one respectively
        :param temporal_downscale: (int) Downscale factor for temporal dimension for spectrum
        """
        # Call super constructor
        super(ECGDataset, self).__init__()
        # Check parameters
        assert isinstance(ecg_leads, List), "ECG leads musst be a list of np.ndarray."
        assert isinstance(ecg_labels, List), "ECG labels musst be a list of strings."
        if augmentation_pipeline is not None:
            assert isinstance(augmentation_pipeline, nn.Module), "Augmentation pipeline must be a torch.nn.Module."
        assert isinstance(spectrogram_length, int) and spectrogram_length > 0, \
            "Spectrogram length must be a positive integer."
        assert isinstance(normalize, bool), "Normalize must be a bool"
        assert isinstance(temporal_downscale, int) and temporal_downscale > 0, \
            "Temporal downscale must be a positive integer"
        # Save parameters
        self.ecg_leads = [torch.from_numpy(data_sample).float() for data_sample in ecg_leads]
        self.ecg_labels = []
        for ecg_label in ecg_labels:
            if ecg_label == "N":
                self.ecg_labels.append(torch.tensor(0, dtype=torch.long))
            elif ecg_label == "O":
                self.ecg_labels.append(torch.tensor(1, dtype=torch.long))
            elif ecg_label == "A":
                self.ecg_labels.append(torch.tensor(2, dtype=torch.long))
            elif ecg_label == "~":
                self.ecg_labels.append(torch.tensor(3, dtype=torch.long))
            else:
                raise RuntimeError("Invalid label value detected!")
        self.augmentation_pipeline = augmentation_pipeline if augmentation_pipeline is not None else nn.Identity()
        self.spectrogram_length = spectrogram_length
        self.normalize = normalize
        self.temporal_downscale = temporal_downscale

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.ecg_leads)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single instance of the dataset
        :param item: (int) Index of the dataset instance to be returned
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) ECG lead, spectrogram, label
        """
        # Get ecg lead, label, and name
        ecg_lead = self.ecg_leads[item][:18000]
        ecg_label = self.ecg_labels[item]
        # Apply augmentations
        ecg_lead = self.augmentation_pipeline(ecg_lead)
        # Normalize signal if utilized
        if self.normalize:
            ecg_lead = (ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08)
        # Compute spectrogram of ecg_lead
        spectrogram, _ = pywt.cwt(ecg_lead.numpy(), range(1, 129), wavelet="morl", )
        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = F.avg_pool1d(spectrogram[None], kernel_size=self.temporal_downscale,
                                   stride=self.temporal_downscale)[0]
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spectrogram_padded = F.pad(spectrogram,
                                   pad=(
                                   0, (self.spectrogram_length // self.temporal_downscale) - spectrogram.shape[-1]))
        # Make mask
        mask = torch.zeros(spectrogram_padded.shape[1])
        mask[((self.spectrogram_length // self.temporal_downscale) - spectrogram.shape[-1]):] = 1
        # Label to one hot encoding
        ecg_label = F.one_hot(ecg_label, num_classes=4)
        return spectrogram_padded.permute(1, 0).float(), mask.bool(), ecg_label


if __name__ == '__main__':
    from wettbewerb import load_references

    ecg_leads, ecg_labels, _, _ = load_references(folder="../data/training/")
    dataset = ECGDataset(ecg_leads, ecg_labels)

    lead, mask, label = dataset[0]

    print(lead.shape, mask.shape, label.shape)

