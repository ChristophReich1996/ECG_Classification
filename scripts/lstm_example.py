import torch
import torch.nn as nn

if __name__ == '__main__':
    input = torch.rand(5, 3, 128)
    lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, bias=True, batch_first=False)
    print(sum([p.numel() for p in lstm.parameters()]))
    output, _ = lstm(input)
    print(output.shape)
