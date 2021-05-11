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
[implementation (cuda extension) from the authors](https://github.com/ml-research/pau).

## Usage

## Results

## Related Work

## References
