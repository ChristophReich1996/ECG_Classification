from typing import Tuple

import torch
import torch.nn as nn


class SoftmaxCrossEntropyLoss(nn.Module):
    """
    This class implements the softmax cross entropy loss.
    """

    def __init__(self, weight: Tuple[float, ...] = (1., 1., 1., 1.)) -> None:
        """
        Constructor methods
        :param weight: (Tuple[float, float, float, float]) Class weights to be applied
        """
        # Call super constructor
        super(SoftmaxCrossEntropyLoss, self).__init__()
        # Check parameter
        assert isinstance(weight, tuple), "Weight parameter must be a tuple of floats."
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
        self.weight = self.weight.to(prediction.device)
        # Calc weighted cross entropy loss
        loss = - (self.weight * label * torch.log_softmax(prediction, dim=-1)).mean()
        return loss


class SoftmaxFocalLoss(nn.Module):
    """
    This class implements the softmax focal loss.
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0) -> None:
        """
        Constructor methods
        :param kwargs: Unused key word arguments
        """
        # Call super constructor
        super(SoftmaxFocalLoss, self).__init__()
        # Check arguments
        assert isinstance(alpha, float), "Alpha parameter must be a float."
        assert isinstance(gamma, float), "Gamma parameter must be a float."
        # Save parameters
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction: torch.Tensor, label: torch.tensor) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) Raw prediction
        :param label: (torch.Tensor) Label as long index
        :return: (torch.Tensor) Loss value
        """
        # Apply softmax activation
        prediction_softmax = torch.softmax(prediction, dim=-1) + 1e-08
        # Compute focal loss
        weight = prediction_softmax
        focal = - self.alpha * weight * torch.log(prediction_softmax)
        loss = (label * focal).sum(dim=-1).mean()
        return loss
