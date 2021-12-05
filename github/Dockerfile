# Use PyTorch 1.8.1 image
FROM nvcr.io/nvidia/pytorch:21.02-py3
# Set maintainer
MAINTAINER creich
# Perform update
RUN ["apt-get", "update"]
# Install ninja
RUN ["apt-get", "install", "-y", "ninja-build"]
# Make PhysioNet folder
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet
# Install Python requirements
RUN pip install --no-deps -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
# Install PAU
WORKDIR /physionet/ecg_classification/pade_activation_unit/cuda
RUN python setup.py install
# Reset working directory
WORKDIR /physionet