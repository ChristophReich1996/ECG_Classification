from typing import Tuple

import torch
from torch.utils.data import Dataset


class PhysioNetDataset(Dataset):
    """
    This class implements the PhysioNet dataset for ECG classification.
    """

    def __init__(self) -> None:
        # Call super constructor
        super(PhysioNetDataset, self).__init__()

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        :return: (int) Length of the dataset
        """
        pass

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single instance of the dataset
        :param item: (int) Index of the dataset instance to be returned
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor])
        """
        pass
