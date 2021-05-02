import torch
import torch.nn as nn


class Accuracy(nn.Module):
    """
    This class implements the accuracy as a nn.Module
    """

    def __init__(self) -> None:
        # Call super constructor
        super(Accuracy, self).__init__()

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) Prediction tensor with the shape [batch size, n classes]
        :param label: (torch.Tensor) Label tensor with the shape [batch size, n classes]
        :return: (torch.Tensor) Accuracy value
        """
        # Threshold prediction with argmax
        prediction_argmax = prediction.argmax(dim=-1)
        # Apply argmax to label
        label = label.argmax(dim=-1)
        # Compute accuracy
        accuracy = (prediction_argmax == label).float().mean()
        return accuracy
