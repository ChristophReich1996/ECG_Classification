import torch

from ecg_classification.pade_activation_unit.utils import PAU
from ecg_classification.pade_activation_unit.pytorch_impl import PADEACTIVATION_Function_based


def main() -> None:
    # Init original PAU
    pau = PAU()
    # Init PAU Pytorch
    pau_pytorch = PADEACTIVATION_Function_based()
    # Load weights
    weights = torch.load("../experiments/13_05_2021__01_37_34ECGCNN_S_physio_net_dataset/models/best_model.pt")
    # Get PAU weights
    weights = {k.split(".")[-1]: v for k, v in weights.items() if k in ["module.linear_layer_1.1.weight_numerator",
                                                                        "module.linear_layer_1.1.weight_denominator"]}
    # Load weights
    pau.load_state_dict(weights)
    pau_pytorch.load_state_dict(weights)
    print(pau.weight_numerator, pau.weight_denominator)
    print(pau_pytorch.weight_numerator, pau_pytorch.weight_denominator)
    # Init input
    input = torch.ones(4, 1, dtype=torch.float)
    # Get PAU output
    pau.cuda()
    output_pau = pau(input.clone().cuda())
    print(output_pau)
    # Get PAU Pytorch output
    output_pau_pytorch = pau_pytorch(input)
    print(output_pau_pytorch)


if __name__ == '__main__':
    main()
