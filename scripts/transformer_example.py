import torch
import torch.nn as nn

if __name__ == '__main__':
    transformer = nn.Transformer(d_model=128, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
                                 dim_feedforward=256, dropout=0.1, activation='relu', custom_encoder=None,
                                 custom_decoder=None)
    src_1 = torch.rand(12, 128)
    src_2 = torch.rand(8, 128)
    src = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.pack_sequence([src_1, src_2]), batch_first=False)[0]
    tgt = torch.rand(4, 2, 128)
    output = transformer(src, tgt)
    print(output.shape)

