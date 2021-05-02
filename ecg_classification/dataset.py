from typing import Tuple, List, Optional

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
                 augmentation_pipeline: Optional[nn.Module] = None, spectrogram_length: int = 80,
                 spectrogram_shape: Tuple[int, int] = (128, 128), ecg_sequence_length: int = 18000,
                 ecg_window_size: int = 256, ecg_step: int = 256 - 32, normalize: bool = True, fs: int = 300) -> None:
        """
        Constructor method
        :param ecg_leads: (List[np.ndarray]) ECG data as list of numpy arrays
        :param ecg_labels: (List[str]) ECG labels as list of strings (N, O, A, ~)
        :param augmentation_pipeline: (Optional[nn.Module]) Augmentation pipeline
        :param spectrogram_length: (int) Fixed spectrogram length (achieved by zero padding)
        :param spectrogram_shape: (Tuple[int, int]) Final size of the spectrogram
        :param ecg_sequence_length: (int) Fixed length of sequence
        :param ecg_window_size: (int) Window size to be applied during unfolding
        :param ecg_step: (int) Step size of unfolding
        :param normalize: (bool) If true signal is normalized to a mean and std of zero and one respectively
        :param fs: (int) Sampling frequency
        """
        # Call super constructor
        super(PhysioNetDataset, self).__init__()
        # Check parameters
        assert isinstance(ecg_leads, List), "ECG leads musst be a list of np.ndarray."
        assert isinstance(ecg_labels, List), "ECG labels musst be a list of strings."
        if augmentation_pipeline is not None:
            assert isinstance(augmentation_pipeline, nn.Module), "Augmentation pipeline must be a torch.nn.Module."
        assert isinstance(spectrogram_length, int) and spectrogram_length > 0, \
            "Spectrogram length must be a positive integer."
        assert isinstance(spectrogram_shape, tuple), "Spectrogram shape must be a tuple of ints."
        assert isinstance(ecg_sequence_length, int) and ecg_sequence_length > 0, \
            "ECG sequence length must be a positive integer."
        assert isinstance(ecg_window_size, int) and ecg_window_size > 0, "ECG window size must be a positive integer."
        assert isinstance(ecg_step, int) and ecg_step > 0 and ecg_step < ecg_window_size, \
            "ECG step must be a positive integer but must be smaller than the window size."
        assert isinstance(normalize, bool), "Normalize must be a bool"
        assert isinstance(fs, int), "Sampling frequency fs must be a int value"
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
        self.ecg_sequence_length = ecg_sequence_length
        self.spectrogram_shape = spectrogram_shape
        self.ecg_window_size = ecg_window_size
        self.ecg_step = ecg_step
        self.normalize = normalize
        self.fs = fs

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
        # Normalize signal if utilized
        if self.normalize:
            ecg_lead = (ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08)
        # Compute spectrogram of ecg_lead
        f, t, spectrogram = scipy.signal.spectrogram(x=ecg_lead.numpy(), fs=self.fs)
        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = torch.log(spectrogram + 1e-05)
        # Pad spectrogram to the desired shape
        spectrogram = F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]),
                            value=0., mode="constant")
        # Reshape spectrogram
        spectrogram = F.interpolate(spectrogram[None, None],
                                    size=self.spectrogram_shape, mode="bicubic", align_corners=False)[0, 0]
        # Pad ecg lead
        ecg_lead = F.pad(ecg_lead, pad=(0, self.ecg_sequence_length - ecg_lead.shape[0]), value=0., mode="constant")
        # Unfold ecg lead
        ecg_lead = ecg_lead.unfold(dimension=-1, size=self.ecg_window_size, step=self.ecg_step)
        # Label to one hot encoding
        ecg_label = F.one_hot(ecg_label, num_classes=4)
        return ecg_lead.float(), spectrogram.unsqueeze(dim=0).float(), ecg_label
