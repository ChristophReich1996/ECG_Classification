from typing import Tuple, List, Optional

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

    def __repr__(self) -> str:
        """
        Returns the name of the class
        :return: (str) Name
        """
        return "Accuracy"

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

    def __init__(self, classes: Optional[Tuple[int, ...]] = None) -> None:
        """
        Constructor method
        :classes (Optional[Tuple[int, ...]]) Classes to be considered
        """
        # Call super constructor
        super(F1, self).__init__()
        # Save parameter
        self.classes = classes

    def __repr__(self) -> str:
        """
        Returns the name of the class
        :return: (str) Name
        """
        return "F1"

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
        # Init list to store the class f1 scores
        class_f1: List[torch.Tensor] = []
        # Iterate over all classes
        for c in (self.classes if self.classes is not None else range(prediction.shape[-1])):
            prediction_ = prediction[..., c]
            label_ = label[..., c]
            # Calc tp, fp, fn
            tp = (label_ * prediction_).sum(dim=-1)
            fp = ((1. - label_) * prediction_).sum(dim=-1)
            fn = (label_ * (1. - prediction_)).sum(dim=-1)
            # Calc prediction and recall
            precision = tp / (tp + fp + 1e-08)
            recall = tp / (tp + fn + 1e-08)
            # Calc F1 score
            f1 = 2. * (precision * recall) / (precision + recall + 1e-08)
            # Save F1 score
            class_f1.append(f1)
        return torch.tensor(class_f1).mean()
