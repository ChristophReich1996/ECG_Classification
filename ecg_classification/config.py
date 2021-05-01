import torch.nn as nn

# Configuration for ECGCNN
ECGCNN_CONFIG = {
    "ecg_encoder_channels": ((80, 128), (128, 256), (256, 256), (256, 128), (128, 32)),
    "spectrogram_encoder_channels": ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
    "latent_vector_features": 256,
    "classes": 4
}

# Configuration for ECGAttNet
ECGAttNet_CONFIG = {
    "ecg_encoder_channels": ((80, 128), (128, 256), (256, 256), (256, 128), (128, 32)),
    "ecg_encoder_spans": (256, 128, 64, 32, 16),
    "spectrogram_encoder_channels": ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
    "spectrogram_encoder_spans": (128, 64, 32, 16, 8),
    "latent_vector_features": 256,
    "classes": 4
}
