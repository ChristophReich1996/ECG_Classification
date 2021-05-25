# ECG Classification

This repository includes the code of the ECG-DualNet for ECG classification. This work was done as part of the 
competition "Wettbewerb künstliche Intelligenz in der Medizin" at TU Darmstadt.

<img src="/github/architecture.png"  alt="architecture" width = 600px height = 320px >

## Installation

All dependencies can be installed by running the following commands:

```shell script
git clone https://github.com/ChristophReich1996/ECG_Classification
cd ECG_Classification
pip install -r requirements.txt
cd ecg_classification/pade_activation_unit/cuda
python setup.py install
cd ../../../
```

The implementation was tested [Gentoo Linux](https://www.gentoo.org/) 5.10.7, Python 3.8.5, and CUDA 11.1. **To perform 
training and validation a CUDA device is needed!** This is due to the PAU implementation, which does not support 
execution on the CPU. The functionality of this repository can not be guaranteed for other system configurations.

If only CUDA 11.0 is available the code can also be executed with PyTorch 1.7.1 and Torchaudio 0.7.2 [see](https://pytorch.org/get-started/previous-versions/).

### Used Implementation

We implement ECG-AttNet with [PyTorch](https://pytorch.org/) 1.8.1 and 
[Torchaudio](https://pytorch.org/audio/stable/index.html) 0.8.1. All required packages can be seen 
in the `requirements.txt` file. For the Pade Activation Unit we adopted the 
[implementation (cuda extension) from the authors](https://github.com/ml-research/pau) [1].

## Results

We achieved the following validation results for our custom training/validation split on the 2017 PhysioNet Challenge 
dataset [2]. Three different training runs are reported. Weights of the best performing model are provided.

| Model | ACC | F1 | # Parameters | best |
| --- | --- | --- | ---: | --- |
| ECG-DualNet S (CNN + LSTM) | 0.8527; 0.8410; 0.8455 | 0.8049; 0.7923; 0.7799 | 1840210 | [weights](experiments/13_05_2021__01_37_34ECGCNN_S_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet M (CNN + LSTM) | 0.8560; 0.8442; 0.8495 | 0.7938; 0.7955; 0.7928 | 4269618 | [weights](experiments/13_05_2021__02_06_41ECGCNN_M_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet L (CNN + LSTM) |  0.8508; 0.8213; 0.8514 | 0.8097; 0.7515; 0.8038 | 6176498 | [weights](experiments/13_05_2021__13_54_12ECGCNN_L_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet XL (CNN + LSTM) | 0.7702; 0.8612; 0.7866 | 0.6899; 0.8164; 0.7162 | 20683122 | [weights](https://studtudarmstadtde-my.sharepoint.com/:u:/g/personal/christoph_reich_stud_tu-darmstadt_de/EXIoAOjpQq1Gh3PPBkwKTDUBTnGVTa8HCjfQHyGmZOVRJg?e=yDCZVV) |
| ECG-DualNet++ S (AxAtt + Trans.) | 0.7323; 0.8174; 0.7912 | 0.6239; 0.7291; 0.7127 | 1785186 | [weights](experiments/13_05_2021__05_35_44ECGAttNet_S_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet++ M (AxAtt + Trans.) | 0.8226; 0.8259; 0.7938 | 0.7544; 0.7730; 0.6947 | 2589714 | [weights](experiments/13_05_2021__06_46_54ECGAttNet_M_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet++ L (AxAtt + Trans.) | 0.8449; 0.8442; 0.8396 | 0.7859; 0.7750; 0.7671 | 3717426 | [weights](experiments/13_05_2021__07_56_52ECGAttNet_L_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet++ XL (AxAtt + Trans.) | 0.8593; 0.8351; 0.8501 | 0.8051; 0.7799; 0.7851 | 8212658 | [weights](experiments/13_05_2021__09_42_13ECGAttNet_XL_physio_net_dataset/models/best_model.pt) |
| ECG-DualNet++ 130M (AxAtt + Trans.) | 0.8475; 0.8534; 0.8462 | 0.7878; 0.7963; 0.7740 | 127833010 | [weights](https://studtudarmstadtde-my.sharepoint.com/:u:/g/personal/christoph_reich_stud_tu-darmstadt_de/EfQSSiciRjBOvUORGYyMEMEBqjD5yjolLFoZIMjywyiKXw?e=z2p7vh) |

Note that for the weights of ECG-DualNet XL an external link is provided.

For training on the Icentia11k dataset [3] we achieved the following results:

| Model | ACC | F1 | # Parameters | best | 20 epochs |
| --- | --- | --- | ---: | --- | --- |
| ECG-DualNet XL (CNN + LSTM) | 0.8989 | 0.4564 | 20683122 | [weights](experiments/20_05_2021__18_32_19ECGAttNet_XL_icentia11k_dataset/models/best_model.pt) | [weights](experiments/20_05_2021__18_32_19ECGAttNet_XL_icentia11k_dataset/models/20.pt) |
| ECG-DualNet++ XL (AxAtt + Trans.) | 0.8899 | 0.4970 | 8212658 | [weights](https://studtudarmstadtde-my.sharepoint.com/:u:/g/personal/christoph_reich_stud_tu-darmstadt_de/EULh87xNGmBAueOrFQCkbxYB9xHukZjmyk4wujNvuw58lA?e=Dbdw86) | [weights](https://studtudarmstadtde-my.sharepoint.com/:u:/g/personal/christoph_reich_stud_tu-darmstadt_de/EVaS1KbQ6fRDo-IJYjOL1GQB5oRkx4IZB5IUWCOWWsoomA?e=3Nr9RD) |

## Usage

To reproduce the presented results simply run (a single GPU is needed):

```shell script
sh run_experiments.sh
```

This script trains all models listed in the table above except ECG-DualNet++ 130M. During training all logs are saved in the experiments folder 
(produced automatically). Most logs are stored in the 
[PyTorch tensor format](https://pytorch.org/docs/stable/generated/torch.load.html) `.pt` and can be loaded by 
`loss:torch.Tensor = torch.load("loss.pt"")`. Network weights are stored as a 
[state dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) and can be loaded 
by `state_dict:Dict[str, torch.Tensor] = torch.load("best_model.pt")`.

To train the biggest ECG-DualNet++ with 130M parameters run:

```shell script
python -W ignore train.py --cuda_devices "0, 1, 2, 3" --epochs 100 --batch_size 24 --physio_net --dataset_path "data/training2017/" --network_config "ECGAttNet_130M" --data_parallel
```

Four GPUs with 16GB are recommended. Reducing the batch size is a possible workaround if limited GPU memory is available.

To reproduce our presented ablation studies run:

```shell script
sh run_ablations.sh
```

To perform pretraining on the Icentia11k dataset [3] run:

```shell script
python -W ignore train.py --cuda_devices "0" --batch_size 100 --dataset_path "/dataset/icentia11k" --icentia11k --network_config "ECGAttNet_XL" --epochs 20
```

Pretraining with a batch size of 100 requres a GPU with at least 32GB. If a batch size of 50 is utilized a 16GB GPU is needed. Batch size can only be set in steps of 50.

To train the pretrained models on the 2017 PhysioNet dataset [2] run:

```shell script
python -W ignore train.py --cuda_devices "0" --epochs 100 --batch_size 24 --physio_net --dataset_path "data/training2017/" --network_config "ECGAttNet_XL" --load_network "experiments/20_05_2021__18_32_19ECGAttNet_XL_icentia11k_dataset/models/20.pt"
```

or

```shell script
python -W ignore train.py --cuda_devices "0" --epochs 100 --batch_size 24 --physio_net --dataset_path "data/training2017/" --network_config "ECGCNN_XL" --load_network "experiments/21_05_2021__12_15_06ECGCNN_XL_icentia11k_dataset/models/20.pt"
```

The challange submission can be reproduced by setting the additional flag `--challange`.

Pleas note that the dataset or model paths as well as the cuda devices might change for different systems!

To run custom training the `train.py` script can be used. This script takes the following commands:

| Argument | Default Value | Info |
| --- | --- | --- |
| `--cuda_devices` | "0" | String of cuda device indexes to be used. Indexes must be separated by a comma. |
| `--no_data_aug` | False | Binary flag. If set no data augmentation is utilized. |
| `--data_parallel` | False | Binary flag. If set data parallel is utilized. |
| `--epochs` | 100 | Number of epochs to perform while training. |
| `--lr` | 1e-03 | Learning rate to be employed. |
| `--physio_net` | False | Binary flag. Utilized PhysioNet dataset instead of default one. |
| `--batch_size` | 24 | Number of epochs to perform while training. |
| `--dataset_path` | False | Path to dataset. |
| `--network_config` | "ECGCNN_M" | Type of network configuration to be utilized. |
| `--load_network` | None | If set given network (state dict) is loaded. |

All network hyperparameters can be found and adjusted in the `ecg_classification\config.py` file.

## Data

The cleaned data of the challenge (also coming from [2]) as well as the publically available PhysioNet [2] samples can be downloaded [here](https://studtudarmstadtde-my.sharepoint.com/:u:/g/personal/christoph_reich_stud_tu-darmstadt_de/Ebl-lX1RfsFLjLWNfdmOaFMBczbye6m_vOYjbhhvFHd7Lg?e=XxcvzV).

The Icentia11k dataset [3] used for pretraining can be downloaded [here](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272).

## References

```bibtex
[1] @article{Molina2019,
        title={Pad\'{e} Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks},
        author={Molina, Alejandro and Schramowski, Patrick and Kersting, Kristian},
        journal={preprint arXiv:1907.06732},
        year={2019}
}
```

```bibtex
[2] @inproceedings{Clifford2017,
        title={{AF Classification from a Short Single Lead ECG Recording: the
        PhysioNet/Computing in Cardiology Challenge 2017}},
        author={Clifford, Gari D and Liu, Chengyu and Moody, Benjamin and Li-wei, H Lehman and Silva, Ikaro and Li, Qiao and Johnson, AE and Mark, Roger G},
        booktitle={2017 Computing in Cardiology (CinC)},
        pages={1--4},
        year={2017},
        organization={IEEE}
}
```

```bibtex
[3] @article{Tan2019,
        title={Icentia11k: An unsupervised representation learning dataset for arrhythmia subtype discovery},
        author={Tan, Shawn and Androz, Guillaume and Chamseddine, Ahmad and Fecteau, Pierre and Courville, Aaron and Bengio, Yoshua and Cohen, Joseph Paul},
        journal={preprint arXiv:1910.09570},
        year={2019}
}
```
