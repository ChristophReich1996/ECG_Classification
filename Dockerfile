# Use PyTorch 1.8.1 image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
# Set maintainer
MAINTAINER creich
# Make PhysioNet folder
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet
# Install Python requirements
RUN pip install -r requirements.txt 
# Reset working directory
WORKDIR /physionet