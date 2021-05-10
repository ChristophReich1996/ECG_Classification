from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchaudio.transforms import Spectrogram


class ECGDataset(Dataset):
    """
    This class implements the ECG dataset for atrial fibrillation classification.
    """

    def __init__(self, ecg_leads: List[np.ndarray], ecg_labels: List[str],
                 augmentation_pipeline: Optional[nn.Module] = None, spectrogram_length: int = 563,
                 ecg_sequence_length: int = 18000, ecg_window_size: int = 256, ecg_step: int = 256 - 32,
                 normalize: bool = True, fs: int = 300, spectrogram_n_fft: int = 64, spectrogram_win_length: int = 64,
                 spectrogram_power: int = 1, spectrogram_normalized: bool = True) -> None:
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
        :param spectrogram_n_fft: (int) FFT size utilized in spectrogram
        :param spectrogram_win_length: (int) Spectrogram window length
        :param spectrogram_power: (int) Power utilized in spectrogram
        :param spectrogram_normalized: (int) If true spectrogram is normalized
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
        assert isinstance(ecg_sequence_length, int) and ecg_sequence_length > 0, \
            "ECG sequence length must be a positive integer."
        assert isinstance(ecg_window_size, int) and ecg_window_size > 0, "ECG window size must be a positive integer."
        assert isinstance(ecg_step, int) and ecg_step > 0 and ecg_step < ecg_window_size, \
            "ECG step must be a positive integer but must be smaller than the window size."
        assert isinstance(normalize, bool), "Normalize must be a bool"
        assert isinstance(fs, int), "Sampling frequency fs must be a int value"
        assert isinstance(spectrogram_n_fft, int) and spectrogram_n_fft > 0, \
            "Spectrogram number of ffts must be an positive integer"
        assert isinstance(spectrogram_win_length, int) and spectrogram_win_length > 0, \
            "Spectrogram window length must be an positive integer"
        assert isinstance(spectrogram_power, int) and spectrogram_power > 0, \
            "Spectrogram power must be an positive integer"
        assert isinstance(spectrogram_normalized, bool), "Spectrogram normalized must be a bool"
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
        self.ecg_window_size = ecg_window_size
        self.ecg_step = ecg_step
        self.normalize = normalize
        self.fs = fs
        self.spectrogram_module = Spectrogram(n_fft=spectrogram_n_fft, win_length=spectrogram_win_length,
                                              hop_length=spectrogram_win_length // 2, power=spectrogram_power,
                                              normalized=spectrogram_normalized)

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
        ecg_lead = self.ecg_leads[item][:self.ecg_sequence_length]
        ecg_label = self.ecg_labels[item]
        # Apply augmentations
        ecg_lead = self.augmentation_pipeline(ecg_lead)
        # Normalize signal if utilized
        if self.normalize:
            ecg_lead = (ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08)
        # Compute spectrogram of ecg_lead
        spectrogram = self.spectrogram_module(ecg_lead)
        spectrogram = torch.log(spectrogram.abs().clamp(min=1e-08))
        # Pad spectrogram to the desired shape
        spectrogram = F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]),
                            value=0., mode="constant").permute(1, 0)
        # Pad ecg lead
        ecg_lead = F.pad(ecg_lead, pad=(0, self.ecg_sequence_length - ecg_lead.shape[0]), value=0., mode="constant")
        # Unfold ecg lead
        ecg_lead = ecg_lead.unfold(dimension=-1, size=self.ecg_window_size, step=self.ecg_step)
        # Label to one hot encoding
        ecg_label = F.one_hot(ecg_label, num_classes=4)
        return ecg_lead.float(), spectrogram.unsqueeze(dim=0).float(), ecg_label


if __name__ == '__main__':
    from wettbewerb import load_references

    ecg_leads, ecg_labels, _, _ = load_references("../data/training/")
    dataset = ECGDataset(ecg_leads=ecg_leads, ecg_labels=ecg_labels)
    ecg_lead, spectrogram, ecg_label = dataset[0]
    print(ecg_lead.shape, spectrogram.shape, ecg_label.shape)
