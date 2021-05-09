from typing import Any, Dict, Tuple, Type, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGTransformer(nn.Module):
    """
    This class implements the ECG transformer for ECG classification.
    """

    def __init__(self) -> None:
        # Call super constructor
        super(ECGTransformer, self).__init__()
        # Init transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
                                                   activation="relu")
        encoder_norm = nn.LayerNorm(normalized_shape=128, elementwise_affine=False)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6,
                                             norm=encoder_norm)
        self.final_mapping = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input = input.permute(1, 0, 2)
        output = self.encoder(input[:1000])
        output = output.mean(dim=0)
        output = self.final_mapping(output)
        if not self.training:
            output = output.softmax(dim=-1)
        return output
