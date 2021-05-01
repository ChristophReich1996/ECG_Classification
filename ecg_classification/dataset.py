from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.signal
import numpy as np


class PhysioNetDataset(Dataset):
    """
    This class implements the PhysioNet dataset for ECG classification.
    """

    def __init__(self, ecg_leads: List[np.ndarray], ecg_labels: List[str],
                 augmentation_pipeline: nn.Module = nn.Identity(), spectrogram_length: int = 80,
                 ecg_sequence_length: int = 18000) -> None:
        """
        Constructor method
        :param ecg_leads: (List[np.ndarray]) ECG data as list of numpy arrays
        :param ecg_labels: (List[str]) ECG labels as list of strings (N, O, A, ~)
        :param augmentation_pipeline: (nn.Module) Augmentation pipeline
        :param spectrogram_length: (int) Fixed spectrogram length (achieved by zero padding)
        :param ecg_sequence_length: (int) Fixed length of sequence
        """
        # Call super constructor
        super(PhysioNetDataset, self).__init__()
        # Check parameters
        assert isinstance(ecg_leads, List), "ECG leads musst be a list of np.ndarray."
        assert isinstance(ecg_labels, List), "ECG labels musst be a list of strings."
        assert isinstance(augmentation_pipeline, nn.Module), "Augmentation pipeline must be a torch.nn.Module."
        assert isinstance(spectrogram_length, int) and spectrogram_length > 0, \
            "Spectrogram length must be a positive integer."
        assert isinstance(ecg_sequence_length, int) and ecg_sequence_length > 0, \
            "ECG sequence length must be a positive integer"
        # Save parameters
        self.ecg_leads = [torch.from_numpy(data_sample) for data_sample in ecg_leads]
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
        self.augmentation_pipeline = augmentation_pipeline
        self.spectrogram_length = spectrogram_length
        self.ecg_sequence_length = ecg_sequence_length

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
        ecg_lead = self.ecg_leads[item]
        ecg_label = self.ecg_labels[item]
        # Apply augmentations
        ecg_lead = self.augmentation_pipeline(ecg_lead)
        # Compute spectrogram of ecg_lead
        _, _, spectrogram = scipy.signal.spectrogram(x=ecg_lead.numpy(), return_onesided=False)
        spectrogram = torch.from_numpy(spectrogram).permute(1, 0)
        # Pad spectrogram to the desired shape
        spectrogram = F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]),
                            value=0., mode="constant")
        # Pad ecg lead
        ecg_lead = F.pad(ecg_lead, pad=(0, self.ecg_sequence_length - ecg_lead.shape[0]), value=0., mode="constant")
        # Label to one hot encoding
        ecg_label = F.one_hot(ecg_label, num_classes=4)
        return ecg_lead.unsqueeze(dim=0).float(), spectrogram.unsqueeze(dim=0).float(), ecg_label
