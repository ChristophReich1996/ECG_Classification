from typing import Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AxialAttention2d(nn.Module):
    """
    This class implements the axial attention operation for 2d volumes.
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int, span: int, groups: int = 8) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param dim: (int) Dimension attention is applied to (0 = height, 1 = width, 2 = depth)
        :param span: (int) Span of attention to be used
        :param groups: (int) Multi head attention groups to be used
        """
        # Call super constructor
        super(AxialAttention2d, self).__init__()
        # Check parameters
        assert (in_channels % groups == 0) and (out_channels % groups == 0), \
            "In and output channels must be a factor of the utilized groups."
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.span = span
        self.groups = groups
        self.group_channels = out_channels // groups
        # Init initial query, key and value mapping
        self.query_key_value_mapping = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_features=2 * out_channels, track_running_stats=True, affine=True)
        )
        # Init output normalization
        self.output_normalization = nn.BatchNorm1d(num_features=2 * out_channels, track_running_stats=True, affine=True)
        # Init similarity normalization
        self.similarity_normalization = nn.BatchNorm2d(num_features=3 * self.groups, track_running_stats=True,
                                                       affine=True)
        # Init embeddings
        self.relative_embeddings = nn.Parameter(torch.randn(2 * self.group_channels, 2 * self.span - 1),
                                                requires_grad=True)
        relative_indexes = torch.arange(self.span, dtype=torch.long).unsqueeze(dim=1) \
                           - torch.arange(self.span, dtype=torch.long).unsqueeze(dim=0) \
                           + self.span - 1
        self.register_buffer("relative_indexes", relative_indexes.view(-1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, h, w, d]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, h, w, d]
        """
        # Reshape input dependent on the dimension to be utilized
        if self.dim == 0:  # Attention over volume height
            input = input.permute(0, 3, 1, 2)  # [batch size, width, in channels, height]
        else:  # Attention over volume width
            input = input.permute(0, 2, 1, 3)  # [batch size, height, in channels, width]
        # Save shapes
        batch_size, dim_1, channels, dim_attention = input.shape
        # Reshape tensor to the shape [batch size * dim 1, channels, dim attention]
        input = input.reshape(batch_size * dim_1, channels, dim_attention).contiguous()
        # Perform query, key and value mapping
        query_key_value = self.query_key_value_mapping(input)
        # Split tensor to get the query, key and value tensors
        query, key, value = query_key_value \
            .reshape(batch_size * dim_1, self.groups, self.group_channels * 2, dim_attention) \
            .split([self.group_channels // 2, self.group_channels // 2, self.group_channels], dim=2)
        # Get all embeddings
        embeddings = self.relative_embeddings.index_select(dim=1, index=self.relative_indexes) \
            .view(2 * self.group_channels, self.span, self.span)
        # Split embeddings
        query_embedding, key_embedding, value_embedding = \
            embeddings.split([self.group_channels // 2, self.group_channels // 2, self.group_channels], dim=0)
        # Apply embeddings to query, key and value
        query_embedded = torch.einsum("bgci, cij -> bgij", query, query_embedding)
        key_embedded = torch.einsum("bgci, cij -> bgij", key, key_embedding)
        # Matmul between query and key
        query_key = torch.einsum("bgci, bgcj -> bgij", query_embedded, key_embedded)
        # Construct similarity map
        similarity = torch.cat([query_key, query_embedded, key_embedded], dim=1)
        # Perform normalization
        similarity = self.similarity_normalization(similarity) \
            .view(batch_size * dim_1, 3, self.groups, dim_attention, dim_attention).sum(dim=1)
        # Apply softmax
        similarity = F.softmax(similarity, dim=3)
        # Calc attention map
        attention_map = torch.einsum("bgij, bgcj->bgci", similarity, value)
        # Calc attention embedded
        attention_map_embedded = torch.einsum("bgij, cij->bgci", similarity, value_embedding)
        # Construct output
        output = torch.cat([attention_map, attention_map_embedded], dim=-1) \
            .view(batch_size * dim_1, 2 * self.out_channels, dim_attention)
        # Final output batch normalization
        output = self.output_normalization(output).view(batch_size, dim_1, self.out_channels, 2,
                                                        dim_attention).sum(dim=-2)
        # Reshape output back to original shape
        if self.dim == 0:  # [batch size, width, depth, in channels, height]
            output = output.permute(0, 2, 3, 1)
        else:  # [batch size, height, depth, in channels, width]
            output = output.permute(0, 2, 1, 3)
        return output


class AxialAttention1d(AxialAttention2d):
    """
    This class implements the axial attention operation for 1d vectors.
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int, span: int, groups: int = 8) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param dim: (int) Dimension attention is applied to (0 = height, 1 = width, 2 = depth)
        :param span: (int) Span of attention to be used
        :param groups: (int) Multi head attention groups to be used
        """
        # Check parameters
        assert dim in [0], "Illegal argument for dimension"
        # Call super constructor
        super(AxialAttention1d, self).__init__(in_channels=in_channels, out_channels=out_channels, dim=dim, span=span,
                                               groups=groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, h]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, h]
        """
        # Reshape tensor to use 2d axial-attention
        input = input.unsqueeze(dim=-1)
        # Perform axial-attention
        output = super().forward(input=input)
        # Reshape output to get desired 2d tensor
        output = output.squeeze(dim=-1)
        return output


class AxialAttention2dBlock(nn.Module):
    """
    This class implements the axial attention block proposed in:
    https://arxiv.org/pdf/2003.07853.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, span: Union[int, Tuple[int, int]],
                 latent_vector_features: int, groups: int = 4, activation: Type[nn.Module] = nn.PReLU,
                 downscale: bool = True, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param latent_vector_features: (int) Number of latent features
        :param span: (Union[int, Tuple[int, int, int]]) Spans to be used in attention layers
        :param groups: (int) Multi head attention groups to be used
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        :param downscale: (bool) If true spatial dimensions of the output tensor are downscaled by a factor of two
        :param dropout: (float) Dropout rate to be utilized
        """
        # Call super constructor
        super(AxialAttention2dBlock, self).__init__()
        # Span to tuple
        span = span if isinstance(span, tuple) else (span, span)
        # Init input mapping
        self.input_mapping_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.input_mapping_norm = ConditionalBatchNormalization(num_features=out_channels,
                                                                latent_vector_features=latent_vector_features,
                                                                track_running_stats=True)
        self.input_mapping_act = activation()
        # Init axial attention mapping
        self.axial_attention_mapping = nn.Sequential(
            AxialAttention2d(in_channels=out_channels, out_channels=out_channels, dim=0, span=span[0], groups=groups),
            AxialAttention2d(in_channels=out_channels, out_channels=out_channels, dim=1, span=span[1], groups=groups)
        )
        # Init dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # Init output mapping
        self.output_mapping_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                             kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.output_mapping_norm = ConditionalBatchNormalization(num_features=out_channels,
                                                                 latent_vector_features=latent_vector_features,
                                                                 track_running_stats=True)
        # Init residual mapping
        self.residual_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                          padding=(0, 0), stride=(1, 1),
                                          bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init pooling layer for downscaling the spatial dimensions
        self.pooling_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input volume tensor of the shape [batch size, in channels, h, w]
        :param latent_vector: (torch.Tensor) Latent vector for CBN
        :return: (torch.Tensor) Output volume tensor of the shape [batch size, out channels, h / 2, w / 2]
        """
        # Perform input mapping
        output = self.input_mapping_act(self.input_mapping_norm(self.input_mapping_conv(input), latent_vector))
        # Perform attention
        output = self.axial_attention_mapping(output)
        # Perform dropout
        output = self.dropout(output)
        # Perform output mapping
        output = self.output_mapping_norm(self.output_mapping_conv(self.pooling_layer(output)), latent_vector)
        # Perform residual mapping
        output = output + self.pooling_layer(self.residual_mapping(input))
        # Perform final activation
        output = self.final_activation(output)
        return output


class AxialAttention1dBlock(nn.Module):
    """
    This class implements the axial attention block proposed in:
    https://arxiv.org/pdf/2003.07853.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, span: int, groups: int = 4,
                 activation: Type[nn.Module] = nn.PReLU, normalization: Tuple[nn.Module] = nn.BatchNorm1d,
                 downscale: bool = True, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param span: (Union[int, Tuple[int, int, int]]) Spans to be used in attention layers
        :param groups: (int) Multi head attention groups to be used
        :param normalization: (Type[nn.Module]) Type of normalization to be used
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        :param downscale: (bool) If true spatial dimensions of the output tensor are downscaled by a factor of two
        :param dropout: (float) Dropout rate to be utilized
        """
        # Call super constructor
        super(AxialAttention1dBlock, self).__init__()
        # Init input mapping
        self.input_mapping_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, padding=0, stride=1, bias=False)
        self.input_mapping_norm = normalization(num_features=out_channels, affine=True, track_running_stats=True)
        self.input_mapping_act = activation()
        # Init axial attention mapping
        self.axial_attention_mapping = AxialAttention1d(in_channels=out_channels, out_channels=out_channels, dim=0,
                                                        span=span, groups=groups)
        # Init dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # Init output mapping
        self.output_mapping_conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                             kernel_size=1, padding=0, stride=1, bias=False)
        self.output_mapping_norm = normalization(num_features=out_channels, affine=True, track_running_stats=True)
        # Init residual mapping
        self.residual_mapping = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                          padding=0, stride=1,
                                          bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init pooling layer for downscaling the spatial dimensions
        self.pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input volume tensor of the shape [batch size, in channels, h
        :return: (torch.Tensor) Output volume tensor of the shape [batch size, out channels, h / 2]
        """
        # Perform input mapping
        output = self.input_mapping_act(self.input_mapping_norm(self.input_mapping_conv(input)))
        # Perform attention
        output = self.axial_attention_mapping(output)
        # Perform dropout
        output = self.dropout(output)
        # Perform output mapping
        output = self.output_mapping_norm(self.output_mapping_conv(self.pooling_layer(output)))
        # Perform residual mapping
        output = output + self.pooling_layer(self.residual_mapping(input))
        # Perform final activation
        output = self.final_activation(output)
        return output
