from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random


class AugmentationPipeline(nn.Module):
    """
    This class implements an augmentation pipeline for ecg leads.
    Inspired by: https://arxiv.org/pdf/2009.04398.pdf
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Config dict
        """
        # Call super constructor
        super(AugmentationPipeline, self).__init__()
        # Save parameters
        self.ecg_sequence_length: int = config["ecg_sequence_length"]
        self.p_aug: float = config["p_aug"]
        self.fs: int = config["fs"]
        self.scale_range: Tuple[float, float] = config["scale_range"]
        self.drop_rate = config["drop_rate"]
        self.interval_length: float = config["interval_length"]
        self.max_shift: int = config["max_shift"]
        self.resample_factors: Tuple[float, float] = config["resample_factors"]
        self.max_offset: float = config["max_offset"]
        self.resampling_points: int = config["resampling_points"]
        self.max_sine_magnitude: float = config["max_sine_magnitude"]
        self.sine_frequency_range: Tuple[float, float] = config["sine_frequency_range"]
        self.kernel: Tuple[float, ...] = config["kernel"]

    def scale(self, ecg_lead: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """
        Scale augmentation:  Randomly scaling data
        :param ecg_lead: (torch.Tensor) ECG leads
        :param scale_range: (Tuple[float, float]) Min and max scaling
        :return: (torch.Tensor) ECG lead augmented
        """
        # Get random scalar
        random_scalar = torch.from_numpy(np.random.uniform(low=scale_range[0], high=scale_range[1], size=1))
        # Apply scaling
        ecg_lead = random_scalar * ecg_lead
        return ecg_lead

    def drop(self, ecg_lead: torch.Tensor, drop_rate: float = 0.025) -> torch.Tensor:
        """
        Drop augmentation: Randomly missing signal values
        :param ecg_lead: (torch.Tensor) ECG leads
        :param drop_rate: (float) Relative umber of samples to be dropped
        :return: (torch.Tensor) ECG lead augmented
        """
        # Estimate number of sample to be dropped
        num_dropped_samples = int(ecg_lead.shape[-1] * drop_rate)
        # Randomly drop samples
        ecg_lead[..., torch.randperm(ecg_lead.shape[-1])[:max(1, num_dropped_samples)]] = 0.
        return ecg_lead

    def cutout(self, ecg_lead: torch.Tensor, interval_length: float = 0.1) -> torch.Tensor:
        """
        Cutout augmentation: Set a random interval signal to 0
        :param ecg_lead: (torch.Tensor) ECG leads
        :param interval_length: (float) Interval lenght to be cut out
        :return: (torch.Tensor) ECG lead augmented
        """
        # Estimate interval size
        interval_size = int(ecg_lead.shape[-1] * interval_length)
        # Get random starting index
        index_start = torch.randint(low=0, high=ecg_lead.shape[-1] - interval_size, size=(1,))
        # Apply cut out
        ecg_lead[index_start:index_start + interval_size] = 0.
        return ecg_lead

    def shift(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000, max_shift: int = 2000) -> torch.Tensor:
        """
        Shift augmentation: Shifts the signal at random
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param max_shift: (int) Max applied shift
        :return: (torch.Tensor) ECG lead augmented
        """
        # Generate shift
        shift = torch.randint(low=0, high=max_shift, size=(1,))
        # Apply shift
        ecg_lead = torch.cat([torch.zeros_like(ecg_lead)[..., :shift], ecg_lead], dim=-1)[:ecg_sequence_length]
        return ecg_lead

    def resample(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000,
                 resample_factors: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        Resample augmentation: Resamples the ecg lead
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param resample_factor: (Tuple[float, float]) Min and max value for resampling
        :return: (torch.Tensor) ECG lead augmented
        """
        # Generate resampling factor
        resample_factor = torch.from_numpy(np.random.uniform(low=resample_factors[0], high=resample_factors[1], size=1))
        # Resample ecg lead
        ecg_lead = F.interpolate(ecg_lead[None, None], size=int(resample_factor * ecg_lead.shape[-1]), mode="linear",
                                 align_corners=False)[0, 0]
        # Apply max lenght if needed
        ecg_lead = ecg_lead[:ecg_sequence_length]
        return ecg_lead

    def random_resample(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000,
                        max_offset: float = 0.03, resampling_points: int = 4) -> torch.Tensor:
        """
        Random resample augmentation: Randomly resamples the signal
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param max_offset: (float) Max resampling offsets between 0 and 1
        :param resampling_points: (int) Initial resampling points
        :return: (torch.Tensor) ECG lead augmented
        """
        # Make coordinates for resampling
        coordinates = 2. * (torch.arange(ecg_lead.shape[-1]).float() / (ecg_lead.shape[-1] - 1)) - 1
        # Make offsets
        offsets = F.interpolate(((2 * torch.rand(resampling_points) - 1) * max_offset)[None, None],
                                size=ecg_lead.shape[-1], mode="linear", align_corners=False)[0, 0]
        # Make grid
        grid = torch.stack([coordinates + offsets, coordinates], dim=-1)[None, None].clamp(min=-1, max=1)
        # Apply resampling
        ecg_lead = F.grid_sample(ecg_lead[None, None, None], grid=grid, mode='bilinear', align_corners=False)[0, 0, 0]
        # Apply max lenght if needed
        ecg_lead = ecg_lead[:ecg_sequence_length]
        return ecg_lead

    def sine(self, ecg_lead: torch.Tensor, max_sine_magnitude: float = 0.2,
             sine_frequency_range: Tuple[float, float] = (0.2, 1.), fs: int = 300) -> torch.Tensor:
        """
        Sine augmentation: Add a sine wave to the entire sample
        :param ecg_lead: (torch.Tensor) ECG leads
        :param max_sine_magnitude: (float) Max magnitude of sine to be added
        :param sine_frequency_range: (Tuple[float, float]) Sine frequency rand
        :param fs: (int) Sampling frequency
        :return: (torch.Tensor) ECG lead augmented
        """
        # Get sine magnitude
        sine_magnitude = torch.from_numpy(np.random.uniform(low=0, high=max_sine_magnitude, size=1))
        # Get sine frequency
        sine_frequency = torch.from_numpy(
            np.random.uniform(low=sine_frequency_range[0], high=sine_frequency_range[1], size=1))
        # Make t vector
        t = torch.arange(ecg_lead.shape[-1]) / float(fs)
        # Make sine vector
        sine = torch.sin(2 * math.pi * sine_frequency * t + torch.rand(1)) * sine_magnitude
        # Apply sine
        ecg_lead = sine + ecg_lead
        return ecg_lead

    def low_pass_filter(self, ecg_lead: torch.Tensor,
                        kernel: Tuple[float, ...] = (1, 6, 15, 20, 15, 6, 1)) -> torch.Tensor:
        """
        Low pass filter: Applied via convolutional filter
        :param ecg_lead: (torch.Tensor) ECG leads
        :return: (torch.Tensor) ECG lead augmented
        """
        # Make kernel
        kernel = torch.tensor(kernel, device=ecg_lead.device, dtype=torch.float32).view(1, 1, -1)
        kernel = kernel / kernel.sum()
        # Apply filtering
        ecg_lead = F.conv1d(input=ecg_lead[None, None], weight=kernel, stride=1, padding=kernel.shape[-1] // 2)[0, 0]
        return ecg_lead

    def forward(self, ecg_lead: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applies augmentation to input tensor
        :param ecg_lead: (torch.Tensor) ECG leads
        :return: (torch.Tensor) ECG lead augmented
        """
        # Apply cut out augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.cutout(ecg_lead, interval_length=self.interval_length)
        # Apply drop augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.drop(ecg_lead, drop_rate=self.drop_rate)
        # Apply random resample augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.random_resample(ecg_lead, ecg_sequence_length=self.ecg_sequence_length,
                                            max_offset=self.max_offset, resampling_points=self.resampling_points)
        # Apply resample augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.resample(ecg_lead, ecg_sequence_length=self.ecg_sequence_length,
                                     resample_factors=self.resample_factors)
        # Apply scale augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.scale(ecg_lead, scale_range=self.scale_range)
        # Apply shift augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.shift(ecg_lead, ecg_sequence_length=self.ecg_sequence_length, max_shift=self.max_shift)
        # Apply sine augmentation
        if random.random() <= self.p_aug:
            ecg_lead = self.sine(ecg_lead, max_sine_magnitude=self.max_sine_magnitude,
                                 sine_frequency_range=self.sine_frequency_range, fs=self.fs)
        # Apply low pass filter
        if random.random() <= self.p_aug:
            ecg_lead = self.low_pass_filter(ecg_lead, kernel=self.kernel)
        return ecg_lead


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from wettbewerb import load_references
    from ecg_classification import AUGMENTATION_PIPELINE_CONFIG

    ecg_leads, ecg_labels, fs, ecg_names = load_references("../data/training/")
    ecg_lead = ecg_leads[21]

    plt.plot(ecg_lead[:4000])
    plt.show()

    ap = AugmentationPipeline(config=AUGMENTATION_PIPELINE_CONFIG)

    output_cutout = ap.low_pass_filter(torch.from_numpy(ecg_lead).clone().float())
    plt.plot(output_cutout[:4000])
    plt.title("Low pass")
    plt.show()

    exit(22)

    output_cutout = ap.cutout(torch.from_numpy(ecg_lead).clone().float())
    plt.plot(output_cutout[:4000])
    plt.title("Cutout")
    plt.show()

    output_drop = ap.drop(torch.from_numpy(ecg_lead).clone().float())
    plt.plot(output_drop[:4000])
    plt.title("Drop")
    plt.show()

    output_scale = ap.scale(torch.from_numpy(ecg_lead).clone().float())
    plt.plot(output_scale[:4000])
    plt.title("Scale")
    plt.show()

    output_shift = ap.shift(torch.from_numpy(ecg_lead).clone().float())
    plt.plot(output_shift[:4000])
    plt.title("Shift")
    plt.show()

    output_sine = ap.sine(torch.from_numpy(ecg_lead).clone().float())
    plt.plot(output_sine[:4000])
    plt.title("Sine")
    plt.show()

    output_resample = ap.resample(torch.from_numpy(ecg_leads[21]).float())
    plt.plot(output_resample[:4000])
    plt.title("Resample")
    plt.show()

    output_random_resample = ap.random_resample(torch.from_numpy(ecg_leads[21]).float())
    plt.plot(output_random_resample[:4000])
    plt.title("Random resample")
    plt.show()

    output_pipeline = ap(torch.from_numpy(ecg_leads[21]).float())
    plt.plot(output_pipeline[:4000])
    plt.title("Full pipeline")
    plt.show()
