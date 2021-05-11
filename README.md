# ECG Classification

This repository includes the code of the ECG-AttNet for ECG classification. This work was done as part of the 
competition "Wettbewerb künstliche Intelligenz in der Medizin" at TU Darmstadt.

## Installation

All dependencies can be installed by running the following commands:

```shell script
git clone https://github.com/ChristophReich1996/ECG_Classification
cd ECG_Classification
pip install -r requirements.txt
cd pade_activation_unit/cuda
python setup.py install
cd ../../
```

The implementation was tested [Gentoo Linux](https://www.gentoo.org/) 5.10.7, Python 3.8.5, and CUDA 11.1. **To perform 
training and validation a CUDA device is needed!** This is due to the PAU implementation, which does not support 
execution on the CPU. The functionality of this repository can not be guaranteed for other system configurations.

### Used Implementation

We implement ECG-AttNet with [PyTorch](https://pytorch.org/) 1.8.1 and 
[Torchaudio](https://pytorch.org/audio/stable/index.html) 0.8.1. All required packages can be seen 
in the `requirements.txt` file. For the pade activation unit we adopted the 
[implementation (cuda extension) from the authors](https://github.com/ml-research/pau) [1].

## Results

We achieved the following validation results for our custom training/validation split on the 2017 PhysioNet Challenge 
dataset [2].

| Model | ACC | F1 | # Parameters |
| --- | --- | --- | --- |
| CNN + LSTM baseline S |  |  |  |
| CNN + LSTM baseline M |  |  |  |
| CNN + LSTM baseline L |  |  |  |
| ECG-AttNet S |  |  |  |
| ECG-AttNet M |  |  |  |
| ECG-AttNet L |  |  |  |

## Usage

To reproduce the presented results simply run:

```shell script
sh run_experiments.sh
```

This script trains all models listed in the table above. During training all logs are saved in the experiments folder 
(produced automatically). Most logs are stored in the 
[PyTorch tensor format](https://pytorch.org/docs/stable/generated/torch.load.html) `.pt` and can be loaded by 
`loss:torch.Tensor = torch.load("loss.pt"")`. Network weights are stored as a 
[state dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) and can be loaded 
by `state_dict:Dict[str, torch.Tensor] = torch.load("best_model.pt")`.

To run custom training runs the `train.py` script can be used. This script takes the following commands:

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