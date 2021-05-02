import torch
import torch.nn as nn


class AugmentationPipeline(nn.Module):
    """
    This class implements an augmentation pipeline for ecg leads.
    """

    def __init__(self) -> None:
        # Call super constructor
        super(AugmentationPipeline, self).__init__()

    def forward(self, ecg_leads: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applies augmentation to input tensor
        :param ecg_leads: (torch.Tensor) ECG leads
        :return: (torch.Tensor) ECG leads augmented
        """
        pass
