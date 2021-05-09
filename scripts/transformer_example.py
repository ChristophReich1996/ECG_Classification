import torch
import torch.nn as nn

if __name__ == '__main__':
    transformer = nn.Transformer(d_model=500, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
                                 dim_feedforward=500, dropout=0.1, activation='relu', custom_encoder=None,
                                 custom_decoder=None)
    transformer.to("cuda:1")
    src = torch.rand(18000 // 4, 1, 500).to("cuda:1")
    tgt = torch.rand(4, 1, 500).to("cuda:1")
    output = transformer(src, tgt)
    print(output.shape)

