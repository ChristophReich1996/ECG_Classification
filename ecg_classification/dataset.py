from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchaudio.transforms import Spectrogram
import os
import gzip
import pickle


class PhysioNetDataset(Dataset):
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
        super(PhysioNetDataset, self).__init__()
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


class Icentia11kDataset(Dataset):
    """
    This class implements the Icentia11k. The data of the Icentia11k will be resampled to a target frequency and
    preprocessed. In the final step the sequences are cropped to a size between 9000 and 18000 samples.
    """

    def __init__(self, path: str, split: List[int], ecg_crop_lengths: Tuple[int, int] = (9000, 18000),
                 original_fs: int = 250, spectrogram_length: int = 563, ecg_sequence_length: int = 18000,
                 ecg_window_size: int = 256, ecg_step: int = 256 - 32, normalize: bool = True, fs: int = 300,
                 spectrogram_n_fft: int = 64, spectrogram_win_length: int = 64, spectrogram_power: int = 1,
                 spectrogram_normalized: bool = True) -> None:
        """
        Constructor method
        :param path: (str) Path to dataset
        :param split: (List[int]) Index of files belonging to the current dataset
        :param ecg_crop_lengths: (Tuple[int, int]) Max and min crop lengths
        :param original_fs: (int) Original sampling frequence
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
        super(Icentia11kDataset, self).__init__()
        # Save parameters
        self.path = path
        self.split = split
        self.ecg_crop_lengths = ecg_crop_lengths
        self.original_fs = original_fs
        self.spectrogram_length = spectrogram_length
        self.ecg_sequence_length = ecg_sequence_length
        self.ecg_window_size = ecg_window_size
        self.ecg_step = ecg_step
        self.normalize = normalize
        self.fs = fs
        self.spectrogram_module = Spectrogram(n_fft=spectrogram_n_fft, win_length=spectrogram_win_length,
                                              hop_length=spectrogram_win_length // 2, power=spectrogram_power,
                                              normalized=spectrogram_normalized)
        # Get paths to samples
        self.paths: List[Tuple[str, str]] = []
        for index in self.split:
            self.paths.append((os.path.join(self.path, "{}_batched.pkl.gz".format(str(index).zfill(5))),
                               os.path.join(self.path, "{}_batched_lbls.pkl.gz".format(str(index).zfill(5)))))
        # Check if files exists
        for file_input, file_label in self.paths:
            assert os.path.isfile(file_input), "File {} not found!".format(file_input)
            assert os.path.isfile(file_label), "File {} not found!".format(file_label)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset
        :return: (int) Dataset length
        """
        return len(self.paths)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method returns one instance of the dataset
        :param item: (int) Item index to be returned
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) ECG lead, spectrogram, label
        """
        # Load inputs and labels
        with gzip.open(self.paths[item][0], "rb") as file:
            inputs = torch.from_numpy(pickle.load(file)).float()
        with gzip.open(self.paths[item][1], "rb") as file:
            labels = pickle.load(file)
        # Make crop indexes
        crop_indexes_low = torch.randint(
            low=0,
            high=int(inputs.shape[-1] - (self.fs / self.original_fs) * max(self.ecg_crop_lengths)),
            size=(inputs.shape[0],))
        crop_indexes_length = torch.randint(
            low=int((self.fs / self.original_fs) * min(self.ecg_crop_lengths)),
            high=int((self.fs / self.original_fs) * max(self.ecg_crop_lengths)),
            size=(inputs.shape[0],))
        # Crop signals
        inputs = [input[low:low + length] for input, low, length in zip(inputs, crop_indexes_low, crop_indexes_length)]
        # Interpolate signals
        inputs = [F.interpolate(input[None, None], scale_factor=self.fs / self.original_fs, mode="linear",
                                align_corners=False)[0, 0] for input in inputs]
        # Compute spectrogram
        spectrograms = [self.spectrogram_module(input).abs().clamp(min=1e-08).log() for input in inputs]
        # Pad inputs
        inputs = torch.stack(
            [F.pad(input, pad=(0, self.ecg_sequence_length - input.shape[0]), value=0., mode="constant")
             for input in inputs], dim=0)
        # Unfold ecg lead
        inputs = inputs.unfold(dimension=-1, size=self.ecg_window_size, step=self.ecg_step)
        print(inputs.shape)
        # Pad spectrograms
        spectrograms = torch.stack(
            [F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]), value=0.,
                   mode="constant").permute(1, 0) for spectrogram in spectrograms], dim=0)
        # Get labels
        return inputs, spectrograms, None


if __name__ == '__main__':
    dataset = Icentia11kDataset(path="E:\\ECG_Data\\icentia11k", split=list(range(10000)))
    print(len(dataset))
    dataset[32]
