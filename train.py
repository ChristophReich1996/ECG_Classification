# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references

from ecg_classification import PhysioNetDataset

if __name__ == '__main__':

    ecg_leads, ecg_labels, fs, ecg_names = load_references("data/training/")
    dataset = PhysioNetDataset(ecg_leads=ecg_leads, ecg_labels=ecg_labels)

    for ecg_lead, spectrogram, label in dataset:
        print(ecg_lead.shape, spectrogram.shape, label.shape)

    exit(22)

