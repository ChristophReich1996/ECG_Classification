from typing import Tuple

import torch
import torch.nn as nn


class SoftmaxCrossEntropyLoss(nn.Module):
    """
    This class implements the softmax cross entropy loss.
    """

    def __init__(self, weight: Tuple[float, float, float, float] = (1., 1., 1., 1.)) -> None:
        """
        Constructor methods
        :param weight: (Tuple[float, float, float, float]) Class weights to be applied
        """
        # Call super constructor
        super(SoftmaxCrossEntropyLoss, self).__init__()
        # Save parameter
        self.weight = torch.tensor(weight, dtype=torch.float32).view(1, -1)

    def forward(self, prediction: torch.Tensor, label: torch.tensor) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) Raw prediction
        :param label: (torch.Tensor) Label as long index
        :return: (torch.Tensor) Loss value
        """
        # Weight to device
        self.weight = self.weight.to(prediction.shape)
        # Calc weighted cross entropy loss
        loss = - (self.weight * label * torch.log_softmax(prediction, dim=-1)).mean()
        return loss
