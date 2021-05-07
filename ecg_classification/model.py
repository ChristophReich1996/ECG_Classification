from typing import Any, Dict, Tuple, Type, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNN(nn.Module):
    """
    This class implements a CNN for ECG classification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Dict with network hyperparameters
        """
        # Call super constructor
        super(ECGCNN, self).__init__()
        # Get parameters
        spectrogram_encoder_channels: Tuple[Tuple[int, int], ...] = config["spectrogram_encoder_channels"]
        classes: int = config["classes"]
        activation: Type[nn.Module] = config["activation"]
        convolution2d: Type[nn.Module] = config["convolution2d"]
        normalization2d: Type[nn.Module] = config["normalization2d"]
        dropout: float = config["dropout"]
        transformer_heads: int = config["transformer_heads"]
        transformer_encoder_layers: int = config["transformer_encoder_layers"]
        transformer_decoder_layers: int = config["transformer_decoder_layers"]
        transformer_feedforward_dims: int = config["transformer_feedforward_dims"]
        transformer_activation: str = config["transformer_activation"]
        # Init transformer
        self.transformer = nn.Transformer(d_model=spectrogram_encoder_channels[-1][-1], nhead=transformer_heads,
                                          num_encoder_layers=transformer_encoder_layers,
                                          num_decoder_layers=transformer_decoder_layers,
                                          dim_feedforward=transformer_feedforward_dims, dropout=dropout,
                                          activation=transformer_activation)
        # Init spectrogram encoder
        self.spectrogram_encoder = nn.ModuleList([Conv2dResidualBlock(in_channels=spectrogram_encoder_channel[0],
                                                                      out_channels=spectrogram_encoder_channel[1],
                                                                      convolution=convolution2d,
                                                                      activation=activation,
                                                                      normalization=normalization2d,
                                                                      dropout=dropout) for
                                                  spectrogram_encoder_channel in spectrogram_encoder_channels])
        # Init final mapping
        self.final_mapping = nn.Sequential(
            nn.Conv1d(in_channels=int((128. / 2 ** (len(spectrogram_encoder_channels))) ** 2), out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=True),
            activation(),
            nn.Linear(in_features=spectrogram_encoder_channels[-1][-1], out_features=classes, bias=True)
        )

    def forward(self, ecg_lead: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param ecg_lead: (torch.Tensor) ECG lead tensor
        :param spectrogram: (torch.Tensor) Spectrogram tensor
        :return: (torch.Tensor) Output prediction
        """
        # Forward pass spectrogram encoder
        for block in self.spectrogram_encoder:
            spectrogram = block(spectrogram)
        # Forward pass transformer
        output = self.transformer(ecg_lead.permute(1, 0, 2),
                                  spectrogram.permute(2, 3, 0, 1).flatten(start_dim=0, end_dim=1)).permute(1, 0, 2)
        # Forward pass final mapping
        output = self.final_mapping(output).squeeze(dim=1)
        # Apply softmax if not training mode
        if self.training:
            return output
        else:
            return output.softmax(dim=-1)


class ECGAttNet(nn.Module):
    """
    This class implements a attention network for ECG classification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Dict with network hyperparameters
        """
        # Call super constructor
        super(ECGAttNet, self).__init__()
        # Get parameters
        spectrogram_encoder_channels: Tuple[Tuple[int, int], ...] = config["spectrogram_encoder_channels"]
        spectrogram_encoder_spans: Tuple[int, ...] = config["spectrogram_encoder_spans"]
        classes: int = config["classes"]
        activation: Type[nn.Module] = config["activation"]
        dropout: float = config["dropout"]
        normalization2d: Type[nn.Module] = config["normalization2d"]
        transformer_heads: int = config["transformer_heads"]
        transformer_encoder_layers: int = config["transformer_encoder_layers"]
        transformer_decoder_layers: int = config["transformer_decoder_layers"]
        transformer_feedforward_dims: int = config["transformer_feedforward_dims"]
        transformer_activation: str = config["transformer_activation"]
        # Init transformer
        self.transformer = nn.Transformer(d_model=spectrogram_encoder_channels[-1][-1], nhead=transformer_heads,
                                          num_encoder_layers=transformer_encoder_layers,
                                          num_decoder_layers=transformer_decoder_layers,
                                          dim_feedforward=transformer_feedforward_dims, dropout=dropout,
                                          activation=transformer_activation)
        # Init spectrogram encoder
        self.spectrogram_encoder = nn.ModuleList()
        for index, (spectrogram_encoder_channel, spectrogram_encoder_span) in \
                enumerate(zip(spectrogram_encoder_channels, spectrogram_encoder_spans)):
            if index in [0, 1]:
                self.spectrogram_encoder.append(
                    Conv2dResidualBlock(
                        in_channels=spectrogram_encoder_channel[0],
                        out_channels=spectrogram_encoder_channel[1],
                        activation=activation,
                        normalization=normalization2d,
                        dropout=dropout)
                )
            else:
                self.spectrogram_encoder.append(
                    AxialAttention2dBlock(
                        in_channels=spectrogram_encoder_channel[0],
                        out_channels=spectrogram_encoder_channel[1],
                        span=spectrogram_encoder_span,
                        activation=activation,
                        normalization=normalization2d,
                        dropout=dropout)
                )
        # Init final mapping
        self.final_mapping = nn.Sequential(
            nn.Conv1d(in_channels=int((128. / 2 ** (len(spectrogram_encoder_channels))) ** 2), out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=True),
            activation(),
            nn.Linear(in_features=spectrogram_encoder_channels[-1][-1], out_features=classes, bias=True)
        )

    def forward(self, ecg_lead: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param ecg_lead: (torch.Tensor) ECG lead tensor
        :param spectrogram: (torch.Tensor) Spectrogram tensor
        :return: (torch.Tensor) Output prediction
        """
        # Forward pass spectrogram encoder
        for block in self.spectrogram_encoder:
            spectrogram = block(spectrogram)
        # Forward pass transformer
        output = self.transformer(ecg_lead.permute(1, 0, 2),
                                  spectrogram.permute(2, 3, 0, 1).flatten(start_dim=0, end_dim=1)).permute(1, 0, 2)
        # Forward pass final mapping
        output = self.final_mapping(output).squeeze(dim=1)
        # Apply softmax if not training mode
        if self.training:
            return output
        else:
            return output.softmax(dim=-1)


class ECGInvNet(ECGCNN):
    """
    This class implements a involution network for ECG classification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Dict with network hyperparameters
        """
        # Call super constructor
        super(ECGInvNet, self).__init__(config)


class Conv2dResidualBlock(nn.Module):
    """
    This class implements a simple residual block with 2d convolutions with conditional batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (1, 1), bias: bool = False,
                 convolution: Type[nn.Module] = nn.Conv2d, activation: Type[nn.Module] = nn.PReLU,
                 pooling: Tuple[nn.Module] = nn.AvgPool2d, normalization: Type[nn.Module] = nn.BatchNorm2d,
                 dropout: float = 0.0) -> None:
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
        :param normalization: (Type[nn.Module]) Normalization to be utilized
        :param dropout: (float) Dropout rate to be applied
        """
        # Call super constructor
        super(Conv2dResidualBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True),
            activation(),
            nn.Dropout(p=dropout),
            convolution(in_channels=out_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True)
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=(1, 1), padding=(0, 0),
                                            bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        self.dropout = nn.Dropout(p=dropout)
        # Init downsampling layer
        self.pooling = pooling(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform skip connection
        output = output + self.residual_mapping(input)
        # Perform final activation
        output = self.final_activation(output)
        # Perform final dropour
        output = self.dropout(output)
        # Perform final downsampling
        return self.pooling(output)


class AxialAttention2d(nn.Module):
    """
    This class implements the axial attention operation for 2d volumes.
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int, span: int, groups: int = 16) -> None:
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


class AxialAttention2dBlock(nn.Module):
    """
    This class implements the axial attention block proposed in:
    https://arxiv.org/pdf/2003.07853.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, span: Union[int, Tuple[int, int]], groups: int = 16,
                 activation: Type[nn.Module] = nn.PReLU, normalization: Type[nn.Module] = nn.BatchNorm2d,
                 downscale: bool = True, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Input channels to be employed
        :param out_channels: (int) Output channels to be utilized
        :param span: (Union[int, Tuple[int, int, int]]) Spans to be used in attention layers
        :param groups: (int) Multi head attention groups to be used
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        :param normalization: (Type[nn.Module]) Type of normalization to be utilized
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
        self.input_mapping_norm = normalization(num_features=out_channels, affine=True, track_running_stats=True)
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
        self.output_mapping_norm = normalization(num_features=out_channels, affine=True, track_running_stats=True)
        # Init residual mapping
        self.residual_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                          padding=(0, 0), stride=(1, 1),
                                          bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init pooling layer for downscaling the spatial dimensions
        self.pooling_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input volume tensor of the shape [batch size, in channels, h, w]
        :return: (torch.Tensor) Output volume tensor of the shape [batch size, out channels, h / 2, w / 2]
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
