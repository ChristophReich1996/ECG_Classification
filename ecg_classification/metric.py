import torch
import torch.nn as nn


class Accuracy(nn.Module):
    """
    This class implements the accuracy as a nn.Module
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
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
        label_argmax = label.argmax(dim=-1)
        # Compute accuracy
        accuracy = (prediction_argmax == label_argmax).float().mean()
        return accuracy


class F1(nn.Module):
    """
    This class implements the F1 score as a nn.Module
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(F1, self).__init__()

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) Prediction tensor with the shape [batch size, n classes]
        :param label: (torch.Tensor) Label tensor with the shape [batch size, n classes]
        :return: (torch.Tensor) Accuracy value
        """
        # Threshold prediction with max
        prediction = (prediction == prediction.max(dim=-1, keepdim=True)[0]).float()
        # Apply max to label
        label = (label == label.max(dim=-1, keepdim=True)[0]).float()
        # Calc tp, fp, fn
        tp = (label * prediction).sum()
        fp = ((1. - label) * prediction).sum()
        fn = (label * (1. - prediction)).sum()
        # Calc prediction and recall
        precision = tp / (tp + fp + 1e-08)
        recall = tp / (tp + fn + 1e-08)
        # Calc F1 score
        f1 = 2 * (precision * recall)  (precision + recall + 1e-08)
        return f1
