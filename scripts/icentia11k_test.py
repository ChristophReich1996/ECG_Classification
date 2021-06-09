from typing import List

import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from ecg_classification import *

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
    # Init model
    # config = ECGAttNet_CONFIG_XL
    # config["classes"] = 7
    # model = ECGAttNet(config=config)
    # model.load_state_dict(
    #     torch.load("../experiments/20_05_2021__18_32_19ECGAttNet_XL_icentia11k_dataset/models/best_model.pt"))
    config = ECGCNN_CONFIG_XL
    config["classes"] = 7
    model = ECGCNN(config=ECGCNN_CONFIG_XL)
    model.load_state_dict(
        torch.load("../experiments/21_05_2021__12_15_06ECGCNN_XL_icentia11k_dataset/models/best_model.pt"))
    # Init dataset
    batch_size = 100
    validation_dataset = DataLoader(
        Icentia11kDataset(path="/home/creich/icentia11k", split=VALIDATION_SPLIT_ICENTIA11K,
                          random_seed=VALIDATION_SEED_ICENTIA11K),
        batch_size=max(1, batch_size // 50), num_workers=20,
        pin_memory=True, drop_last=False, shuffle=False, collate_fn=icentia11k_dataset_collate_fn)
    # Init validation metrics
    validation_metrics = (F1(), Accuracy())
    # Network into eval mode
    model.eval()
    # Network to device
    model.to("cuda")
    # Init lists to store all labels and predictions
    predictions: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    # Init progress bar
    progress_bar = tqdm(total=len(validation_dataset))
    # Validation loop
    with torch.no_grad():
        for batch in validation_dataset:
            # Update progress bar
            progress_bar.update(n=1)
            # Unpack batch
            ecg_leads, spectrogram, labels_ = batch
            # Data to device
            ecg_leads = ecg_leads.to("cuda")
            spectrogram = spectrogram.to("cuda")
            labels_ = labels_.to("cuda")
            # Make prediction
            predictions_ = model(ecg_leads, spectrogram)
            # Save predictions and labels
            predictions.append(predictions_)
            labels.append(labels_)
    # Close progress bar
    progress_bar.close()
    # Pack predictions
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    # Compute metrics
    for metric in validation_metrics:
        metric_value = metric(predictions, labels)
        print(metric, metric_value)
