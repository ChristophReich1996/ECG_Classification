from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ecg_classification import *


def predict_labels(ecg_leads: List[np.ndarray], fs: int, ecg_names: List[str],
                   use_pretrained: bool = False, two_classes: bool = True) -> List[Tuple[str, str]]:
    """
    Function to produce predictions
    :param ecg_leads: (List[np.ndarray]) ECG leads as a list of numpy arrays
    :param fs: (int) Sampling frequency
    :param ecg_names: (List[str]) List of strings with name of each ecg lead
    :param use_pretrained: (bool) If true pre-trained (trained!) model is used
    :param two_classes: (bool) If true model for two classes is utilized else four class model is used
    :return: (List[Tuple[str, str]]) List of tuples including name and prediction
    """
    # Init model
    config = ECGCNN_CONFIG_XL
    config["classes"] = 2 if two_classes else config["classes"]
    network = ECGCNN(config=config)
    # Train model if utilized
    if not use_pretrained:
        pass
    # Load model
    else:
        if two_classes:
            try:
                state_dict = torch.load("experiments/"
                                        "15_06_2021__13_28_55ECGCNN_XL_physio_net_dataset_challange_two_classes/"
                                        "models/best_model.pt")
            except FileNotFoundError as exception:
                print("State dict not found. Download the state dict of ECG-DualNet XL (two class, challange). "
                      "Link in README. Put the state dict into the relative directory "
                      "15_06_2021__13_28_55ECGCNN_XL_physio_net_dataset_challange_two_classes/models/")
                exit(1904)
        else:
            try:
                state_dict = torch.load("experiments/25_05_2021__02_02_11ECGCNN_XL_physio_net_dataset_challange/"
                                        "models/best_model.pt")
            except FileNotFoundError as exception:
                print("State dict not found. Download the state dict of ECG-DualNet XL (four class, challange). "
                      "Link in README. Put the state dict into the relative directory "
                      "experiments/25_05_2021__02_02_11ECGCNN_XL_physio_net_dataset_challange/models/")
                exit(1904)
        # Apply state dict
        network.load_state_dict(state_dict)
    # Init dataset for prediction
    dataset = PhysioNetDataset(ecg_leads=ecg_leads, ecg_labels=["A"] * len(ecg_leads), fs=fs,
                               augmentation_pipeline=None, two_classes=two_classes)
    dataset = DataLoader(dataset=dataset, batch_size=1, num_workers=0, pin_memory=False, drop_last=False, shuffle=False)
    # Make prediction
    return _predict(network=network, dataset=dataset, ecg_names=ecg_names, two_classes=two_classes)


def _train(network: Union[nn.Module, nn.DataParallel]) -> Union[nn.Module, nn.DataParallel]:
    """
    Private function which trains the given model
    :param network: (Union[nn.Module, nn.DataParallel]) Model to be trained
    :return: (Union[nn.Module, nn.DataParallel]) Trained model
    """
    pass


@torch.no_grad()
def _predict(network: Union[nn.Module, nn.DataParallel], dataset: DataLoader, ecg_names: List[str],
             two_classes: bool) -> List[Tuple[str, str]]:
    """
    Private function to make predictions
    :param network: (Union[nn.Module, nn.DataParallel]) Trained model
    :param dataset: (DataLoader) Dataset to be predicted
    :param ecg_names: (List[str]) Name of each sample
    :param two_classes: (bool) If true only two classes are utilized
    :return: (List[Tuple[str, str]]) List of tuples including name and prediction
    """
    # Init list to store predictions
    predictions: List[Tuple[str, str]] = []
    # Network to cuda
    network.cuda()
    # Network into eval mode
    network.eval()
    # Init progress bar
    progress_bar = tqdm(total=len(dataset))
    # Prediction loop
    for name, data in zip(ecg_names, dataset):
        # Update progress bar
        progress_bar.update(n=1)
        # Unpack data
        ecg_lead, spectrogram, _ = data
        # Data to cuda
        ecg_lead = ecg_lead.cuda()
        spectrogram = spectrogram.cuda()
        # Make prediction
        prediction = network(ecg_lead, spectrogram)
        # Threshold prediction
        prediction = prediction.argmax(dim=-1)
        # Construct prediction
        predictions.append((name, _get_prediction_name(prediction=prediction, two_classes=two_classes)))
    # Close progress bar
    progress_bar.close()
    return predictions


def _get_prediction_name(prediction: torch.Tensor, two_classes: bool) -> str:
    """
    Function produces string prediction from raw class prediction
    :param prediction: (torch.Tensor) Prediction of the shape [batch size = 1]
    :param two_classes: (bool) If true two class case is utilized
    :return: (str) String including the class name
    """
    # Check batch size
    assert prediction.shape[0] == 1, "Only a batch size of one is supported."
    # Two class case
    if two_classes:
        if int(prediction.item()) == 0:
            return "N"
        elif int(prediction.item()) == 1:
            return "A"
        else:
            raise RuntimeError("Wrong prediction encountered")
    # Four class case
    if int(prediction.item()) == 0:
        return "N"
    elif int(prediction.item()) == 1:
        return "O"
    elif int(prediction.item()) == 2:
        return "A"
    elif int(prediction.item()) == 3:
        return "~"
    else:
        raise RuntimeError("Wrong prediction encountered")
