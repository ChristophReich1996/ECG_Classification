from argparse import ArgumentParser
import os

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--cuda_devices", default="0, 1, 2, 3", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--cpu", default=False, action="store_true",
                    help="Binary flag. If set all operations are performed on the CPU.")
parser.add_argument("--no_data_aug", default=False, action="store_true",
                    help="Binary flag. If set no data augmentation is utilized.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If set data parallel is utilized.")
parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--lr", default=1e-05, type=float,
                    help="Learning rate to be employed.")
parser.add_argument("--batch_size", default=16, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--dataset_path", default="data/training/", type=str,
                    help="Path to dataset")
# Get arguments
args = parser.parse_args()

# Set device type
device = "cpu" if args.cpu else "cuda"

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import torch
import torch_optimizer
from torch.utils.data import DataLoader

from wettbewerb import load_references
from ecg_classification import *

if __name__ == '__main__':
    # Init network
    network = ECGTransformer()

    # Init data logger
    data_logger = Logger()

    # Print network parameters
    print("# parameters:", sum([p.numel() for p in network.parameters()]))

    # Init data parallel if utlized
    network = torch.nn.DataParallel(network)

    # Init optimizer
    optimizer = torch.optim.Adam(params=network.parameters(), lr=args.lr)

    # Init learning rate schedule
    learning_rate_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[1 * args.epochs // 4, 2 * args.epochs // 4, 3 * args.epochs // 4], gamma=0.1)

    # Init datasets
    ecg_leads, ecg_labels, fs, ecg_names = load_references(args.dataset_path)
    training_dataset = DataLoader(
        ECGDataset(ecg_leads=[ecg_leads[index] for index in TRAINING_SPLIT],
                   ecg_labels=[ecg_labels[index] for index in TRAINING_SPLIT],
                   augmentation_pipeline=None if args.no_data_aug else AugmentationPipeline(
                       AUGMENTATION_PIPELINE_CONFIG)),
        batch_size=args.batch_size, num_workers=min(args.batch_size, 20), pin_memory=True,
        drop_last=False, shuffle=True)
    validation_dataset = DataLoader(
        ECGDataset(ecg_leads=[ecg_leads[index] for index in VALIDATION_SPLIT],
                   ecg_labels=[ecg_labels[index] for index in VALIDATION_SPLIT],
                   augmentation_pipeline=None),
        batch_size=args.batch_size, num_workers=min(args.batch_size, 20), pin_memory=True,
        drop_last=False, shuffle=False)

    # Init model wrapper
    model_wrapper = ModelWrapper(network=network,
                                 optimizer=optimizer,
                                 loss_function=SoftmaxCrossEntropyLoss(
                                     weight=(0.40316665, 0.7145, 0.91316664, 0.96916664)),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 data_logger=data_logger,
                                 learning_rate_schedule=learning_rate_schedule,
                                 device=device)

    # Perform training
    with torch.autograd.detect_anomaly():
        model_wrapper.train(epochs=args.epochs)
