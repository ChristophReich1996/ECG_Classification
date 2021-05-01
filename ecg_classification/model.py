from typing import Tuple, Type

import torch
import torch.nn as nn


class ECGCNN(nn.Module):
    """
    This class implements a CNN for ECG classification.
    """
    pass


class ECGAttNet(nn.Module):
    """
    This class implements a attention network for ECG classification.
    """
    pass


class Conv1dResidualBlock(nn.Module):
    """
    This class implements a simple residal block with 1d convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride=1, padding: int = 1,
                 bias: bool = False, convolution: Type[nn.Conv1d] = nn.Conv1d,
                 normalization: Type[nn.Module] = nn.BatchNorm1d, activation: Type[nn.Module] = nn.PReLU,
                 pooling: Tuple[nn.Module] = nn.AvgPool1d) -> None:
        # Call super constructor
        super(Conv1dResidualBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True),
            activation(),
            convolution(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True),
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                            padding=0, bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init downsampling layer
        self.pooling = pooling(kernel_size=2, stride=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor [batch size, in channels, height]
        :return: (torch.Tensor) Output tensor
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform skip connection
        output = output + self.residual_mapping(input)
        # Perform final activation
        output = self.final_activation(output)
        # Perform final downsampling
        return self.pooling(output)


class Conv2dResidualBlock(nn.Module):
    """
    This class implements a simple residal block with 1d convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (1, 1),
                 bias: bool = False, convolution: Type[nn.Conv2d] = nn.Conv2d,
                 normalization: Type[nn.Module] = nn.BatchNorm2d, activation: Type[nn.Module] = nn.PReLU,
                 pooling: Tuple[nn.Module] = nn.AvgPool2d) -> None:
        # Call super constructor
        super(Conv2dResidualBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True),
            activation(),
            convolution(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True),
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                            padding=0, bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init downsampling layer
        self.pooling = pooling(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor [batch size, in channels, height]
        :return: (torch.Tensor) Output tensor
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform skip connection
        output = output + self.residual_mapping(input)
        # Perform final activation
        output = self.final_activation(output)
        # Perform final downsampling
        return self.pooling(output)
