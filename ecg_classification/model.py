from typing import Tuple, Type

import torch
import torch.nn as nn


class ECGCNN(nn.Module):
    """
    This class implements a CNN for ECG classification.
    """

    def __init__(self,
                 ecg_encoder_channels: Tuple[Tuple[int, int]] = (
                         (80, 128), (128, 256), (256, 256), (256, 128), (128, 32)),
                 spectrogram_encoder_channels: Tuple[Tuple[int, int]] = (
                         (1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
                 classes: int = 4) -> None:
        # Call super constructor
        super(ECGCNN, self).__init__()
        # Init ecg encoder
        self.ecg_encoder = nn.Sequential(
            *[Conv1dResidualBlock(in_channels=ecg_encoder_channel[0], out_channels=ecg_encoder_channel[1]) for
              ecg_encoder_channel in ecg_encoder_channels])
        # Init spectrogram encoder
        self.spectrogram_encoder = nn.ModuleList([Conv2dResidualBlock(in_channels=spectrogram_encoder_channel[0],
                                                                      out_channels=spectrogram_encoder_channel[1]) for
                                                  spectrogram_encoder_channel in spectrogram_encoder_channels])
        # Init final linear layer
        self.linear_layer = nn.Linear(
            in_features=(128 // 2 ** (len(spectrogram_encoder_channels))) ** 2 * spectrogram_encoder_channels[-1][-1],
            out_features=classes, bias=True)

    def forward(self, ecg_lead: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param ecg_lead: (torch.Tensor) ECG lead tensor
        :param spectrogram: (torch.Tensor) Spectrogram tensor
        :return: (torch.Tensor) Output prediction
        """
        # Encode ECG lead
        latent_vector = self.ecg_encoder(ecg_lead).flatten(start_dim=1)
        # Forward pass spectrogram encoder
        for block in self.spectrogram_encoder:
            spectrogram = block(spectrogram, latent_vector)
        # Final linear layer
        output = self.linear_layer(spectrogram.flatten(start_dim=1))
        # Apply softmax if not training mode
        if self.training:
            return output
        else:
            return output.softmax(dim=-1)


class ECGAttNet(nn.Module):
    """
    This class implements a attention network for ECG classification.
    """

    def __init__(self) -> None:
        # Call super constructor
        super(ECGAttNet, self).__init__()


class Conv1dResidualBlock(nn.Module):
    """
    This class implements a simple residal block with 1d convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride=1, padding: int = 1,
                 bias: bool = False, convolution: Type[nn.Conv1d] = nn.Conv1d,
                 normalization: Type[nn.Module] = nn.BatchNorm1d, activation: Type[nn.Module] = nn.PReLU,
                 pooling: Tuple[nn.Module] = nn.AvgPool1d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (int) Kernel size to be used in convolution
        :param stride: (int) Stride factor to be used in convolution
        :param padding: (int) Padding to be used in convolution
        :param bias: (int) If true bias is utilized in each convolution
        :param convolution: (Type[nn.Conv1d]) Type of convolution to be utilized
        :param normalization: (Type[nn.Module]) Type of normalization to be utilized
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        :param pooling: (Type[nn.Module]) Type of pooling layer to be utilized
        """
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
    This class implements a simple residual block with 2d convolutions with conditional batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, latent_vector_features: int = 256,
                 kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1), bias: bool = False, convolution: Type[nn.Conv2d] = nn.Conv2d,
                 activation: Type[nn.Module] = nn.PReLU, pooling: Tuple[nn.Module] = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param latent_vector_features: (int) Feature size of latent tensor for CBN
        :param kernel_size: (Tuple[int, int]) Kernel size to be used in convolution
        :param stride: (Tuple[int, int]) Stride factor to be used in convolution
        :param padding: (Tuple[int, int]) Padding to be used in convolution
        :param bias: (int) If true bias is utilized in each convolution
        :param convolution: (Type[nn.Conv2d]) Type of convolution to be utilized
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        :param pooling: (Type[nn.Module]) Type of pooling layer to be utilized
        """
        # Call super constructor
        super(Conv2dResidualBlock, self).__init__()
        # Init main mapping

        self.main_mapping_conv_1 = convolution(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size, stride=stride,
                                               padding=padding, bias=bias)
        self.main_mapping_norm_1 = ConditionalBatchNormalization(num_features=out_channels,
                                                                 latent_vector_features=latent_vector_features)
        self.main_mapping_act_1 = activation()
        self.main_mapping_conv_2 = convolution(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=kernel_size, stride=stride,
                                               padding=padding, bias=bias)
        self.main_mapping_norm_2 = ConditionalBatchNormalization(num_features=out_channels,
                                                                 latent_vector_features=latent_vector_features)
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                            padding=0, bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init downsampling layer
        self.pooling = pooling(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, input: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor [batch size, in channels, height, width]
        :param latent_vector: (torch.Tensor) Latent vector for CBN
        :return: (torch.Tensor) Output tensor
        """
        # Perform main mapping
        output = self.main_mapping_conv_1(input)
        output = self.main_mapping_norm_1(output, latent_vector)
        output = self.main_mapping_act_1(output)
        output = self.main_mapping_conv_2(output)
        output = self.main_mapping_norm_2(output, latent_vector)
        # Perform skip connection
        output = output + self.residual_mapping(input)
        # Perform final activation
        output = self.final_activation(output)
        # Perform final downsampling
        return self.pooling(output)


class ConditionalBatchNormalization(nn.Module):
    """
    Implementation of conditional batch normalization.
    https://arxiv.org/pdf/1707.00683.pdf
    """

    def __init__(self, num_features: int, latent_vector_features: int, track_running_stats: bool = True) -> None:
        """
        Constructor method
        :param num_features: (int) Number of input feautres
        :param latent_vector_features: (int) Number of latent feautres
        :param track_running_stats: (bool) If true running statistics are tracked
        """
        # Call super constructor
        super(ConditionalBatchNormalization, self).__init__()
        # Init batch normalization layer
        self.batch_normalization = nn.BatchNorm2d(num_features=num_features, track_running_stats=track_running_stats,
                                                  affine=True)
        # Init linear mapping
        self.linear_mapping = nn.Linear(in_features=latent_vector_features, out_features=2 * num_features, bias=False)

    def forward(self, input: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param latent_vector: (torch.Tensor) Input latent vector
        :return: (torch.Tensor) Normalized output vector
        """
        # Normalize input
        output = self.batch_normalization(input)
        # Predict parameters
        scale, bias = self.linear_mapping(latent_vector).chunk(chunks=2, dim=-1)
        # Apply parameters
        output = scale.view(1, -1, 1, 1) * output + bias.view(1, -1, 1, 1)
        return output
