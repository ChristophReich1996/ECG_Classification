# -*- coding: utf-8 -*-
"""
This script tests the trained model
Authors: Christoph Hoog Antink, Maurice Rohr and Christoph Reich
"""
from typing import List, Tuple

import numpy as np


def predict_labels(ecg_leads: List[np.ndarray], fs: float, ecg_names: List[str],
                   use_pretrained: bool = False, two_classes: bool = True) -> List[Tuple[str, str]]:
    """
    Function to produce predictions
    :param ecg_leads: (List[np.ndarray]) ECG leads as a list of numpy arrays
    :param fs: (float) Sampling frequency
    :param ecg_names: (List[str]) List of strings with name of each ecg lead
    :param use_pretrained: (bool) If true pre-trained (trained!) model is used
    :param two_classes: (bool) If true model for two classes is utilized else four class model is used
    :return: (List[Tuple[str, str]]) List of tuples including name and prediction
    """
    pass
