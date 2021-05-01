import torch.nn as nn
from involution import Involution2d

# Configuration for ECGCNN S
ECGCNN_CONFIG_S = {
    "ecg_encoder_channels": ((80, 128), (128, 128), (128, 128), (128, 64), (64, 32)),
    "spectrogram_encoder_channels": ((1, 16), (16, 16), (16, 32), (32, 64), (64, 256)),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "convolution1d": nn.Conv1d,
    "convolution2d": nn.Conv2d,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGCNN M
ECGCNN_CONFIG_M = {
    "ecg_encoder_channels": ((80, 128), (128, 256), (256, 256), (256, 128), (128, 32)),
    "spectrogram_encoder_channels": ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "convolution1d": nn.Conv1d,
    "convolution2d": nn.Conv2d,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGCNN L
ECGCNN_CONFIG_L = {
    "ecg_encoder_channels": ((80, 128), (128, 512), (512, 512), (512, 128), (128, 32)),
    "spectrogram_encoder_channels": ((1, 32), (32, 64), (64, 128), (128, 256), (256, 256)),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "convolution1d": nn.Conv1d,
    "convolution2d": nn.Conv2d,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGAttNet S
ECGAttNet_CONFIG_S = {
    "ecg_encoder_channels": ((80, 128), (128, 128), (128, 128), (128, 64), (64, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 16), (16, 16), (16, 32), (32, 64), (64, 256)),
    "spectrogram_encoder_spans": (128, 64, 32, 16, 8),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGAttNet M
ECGAttNet_CONFIG_M = {
    "ecg_encoder_channels": ((80, 128), (128, 256), (256, 256), (256, 128), (128, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
    "spectrogram_encoder_spans": (128, 64, 32, 16, 8),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGAttNet L
ECGAttNet_CONFIG_L = {
    "ecg_encoder_channels": ((80, 128), (128, 512), (512, 512), (512, 128), (128, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 32), (32, 64), (64, 128), (128, 256), (256, 256)),
    "spectrogram_encoder_spans": (128, 64, 32, 16, 8),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGInvNet S
ECGInvNet_CONFIG_S = {
    "ecg_encoder_channels": ((80, 128), (128, 128), (128, 128), (128, 64), (64, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 16), (16, 16), (16, 32), (32, 64), (64, 256)),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "convolution2d": Involution2d,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGInvNet M
ECGInvNet_CONFIG_M = {
    "ecg_encoder_channels": ((80, 128), (128, 256), (256, 256), (256, 128), (128, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "convolution2d": Involution2d,
    "normalization1d": nn.BatchNorm1d,
}

# Configuration for ECGInvNet L
ECGInvNet_CONFIG_L = {
    "ecg_encoder_channels": ((80, 128), (128, 512), (512, 512), (512, 128), (128, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 32), (32, 64), (64, 128), (128, 256), (256, 256)),
    "latent_vector_features": 256,
    "classes": 4,
    "activation": nn.PReLU,
    "convolution2d": Involution2d,
    "normalization1d": nn.BatchNorm1d,
}
