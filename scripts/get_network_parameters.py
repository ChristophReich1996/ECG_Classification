from ecg_classification import *

if __name__ == '__main__':
    # CNN parameters
    for model_config in [ECGCNN_CONFIG_S, ECGCNN_CONFIG_M, ECGCNN_CONFIG_L, ECGCNN_CONFIG_XL]:
        model = ECGCNN(config=model_config)
        print(sum([p.numel() for p in model.parameters()]))
    # AttNet parameters
    for model_config in [ECGAttNet_CONFIG_S, ECGAttNet_CONFIG_M, ECGAttNet_CONFIG_L, ECGAttNet_CONFIG_XL]:
        model = ECGAttNet(config=model_config)
        print(sum([p.numel() for p in model.parameters()]))
