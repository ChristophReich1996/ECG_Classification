import torch

from ecg_classification import *
from fvcore.nn import FlopCountAnalysis

if __name__ == '__main__':
    # CNN parameters
    for model_config in [ECGCNN_CONFIG_S, ECGCNN_CONFIG_M, ECGCNN_CONFIG_L, ECGCNN_CONFIG_XL]:
        model = ECGCNN(config=model_config)
        print(sum([p.numel() for p in model.parameters()]))
        flops = FlopCountAnalysis(model, (torch.rand(1, 80, 256), torch.rand(1, 1, 563, 33)))
        print(flops.by_module_and_operator())
        print("Flops:", flops.total() / 1000000000)
    # AttNet parameters
    for model_config in [ECGAttNet_CONFIG_S, ECGAttNet_CONFIG_M, ECGAttNet_CONFIG_L, ECGAttNet_CONFIG_XL, ECGAttNet_CONFIG_130M]:
        model = ECGAttNet(config=model_config)
        print(sum([p.numel() for p in model.parameters()]))
        flops = FlopCountAnalysis(model, (torch.rand(1, 80, 256), torch.rand(1, 1, 563, 33)))
        print(flops.by_module_and_operator())
        print("Flops:", flops.total() / 1000000000)
