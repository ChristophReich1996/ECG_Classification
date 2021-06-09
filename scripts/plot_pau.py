import torch
import matplotlib.pyplot as plt
from ecg_classification.pade_activation_unit.utils import PAU

if __name__ == '__main__':
    # Load state dict
    state_dict = torch.load("../experiments/20_05_2021__18_32_19ECGAttNet_XL_icentia11k_dataset/models/best_model.pt")
    # state_dict = torch.load("../experiments/08_06_2021__17_14_58ECGAttNet_XL_physio_net_dataset_pretrained/models/best_model.pt")
    # Get keys of activation weights
    pau_keys = [key for key in state_dict.keys() if "weight_numerator" in key or "weight_denominator" in key]
    pau_keys = list(zip(pau_keys[0::2], pau_keys[1::2]))
    # Plot PAUs
    for pau_key in pau_keys:
        # Init PAU
        pau = PAU()
        pau.cuda()
        pau.weight_numerator = torch.nn.Parameter(state_dict[pau_key[0]])
        pau.weight_denominator = torch.nn.Parameter(state_dict[pau_key[1]])
        # Plot PAU
        with torch.no_grad():
            x = torch.linspace(start=-15, end=15, steps=1000, device="cuda")
            y = pau(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy())
        plt.grid()
        plt.title(pau_key[0])
        plt.show()
