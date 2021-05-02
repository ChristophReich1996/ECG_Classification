from argparse import ArgumentParser
import os

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--cuda_devices", default="0, 1, 2, 3", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--cpu", default=False, action="store_true",
                    help="Binary flag. If set all operations are performed on the CPU.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If set data parallel is utilized.")
parser.add_argument("--epochs", default=40, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--lr", default=1e-03, type=float,
                    help="Learning rate to be employed.")
parser.add_argument("--batch_size", default=12, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--dataset_path", default="data/training/", type=str,
                    help="Path to dataset")
parser.add_argument("--network_config", default="ECGCNN_M", type=str,
                    choices=["ECGCNN_S", "ECGCNN_M", "ECGCNN_L", "ECGAttNet_S", "ECGAttNet_M", "ECGAttNet_L",
                             "ECGInvNet_S", "ECGInvNet_M", "ECGInvNet_L"],
                    help="Type of network configuration to be utilized.")
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
    if args.network_config == "ECGCNN_S":
        network = ECGCNN(config=ECGCNN_CONFIG_S)
        data_logger = Logger(experiment_path_extension="ECGCNN_S")
    elif args.network_config == "ECGCNN_M":
        network = ECGCNN(config=ECGCNN_CONFIG_M)
        data_logger = Logger(experiment_path_extension="ECGCNN_M")
    elif args.network_config == "ECGCNN_L":
        network = ECGCNN(config=ECGCNN_CONFIG_L)
        data_logger = Logger(experiment_path_extension="ECGCNN_L")
    elif args.network_config == "ECGAttNet_S":
        network = ECGAttNet(config=ECGAttNet_CONFIG_S)
        data_logger = Logger(experiment_path_extension="ECGAttNet_S")
    elif args.network_config == "ECGAttNet_M":
        network = ECGAttNet(config=ECGAttNet_CONFIG_M)
        data_logger = Logger(experiment_path_extension="ECGAttNet_M")
    elif args.network_config == "ECGAttNet_L":
        network = ECGAttNet(config=ECGAttNet_CONFIG_L)
        data_logger = Logger(experiment_path_extension="ECGAttNet_L")
    elif args.network_config == "ECGInvNet_S":
        network = ECGInvNet(config=ECGInvNet_CONFIG_S)
        data_logger = Logger(experiment_path_extension="ECGInvNet_S")
    elif args.network_config == "ECGInvNet_M":
        network = ECGInvNet(config=ECGInvNet_CONFIG_M)
        data_logger = Logger(experiment_path_extension="ECGInvNet_M")
    else:
        network = ECGInvNet(config=ECGInvNet_CONFIG_L)
        data_logger = Logger(experiment_path_extension="ECGInvNet_L")

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
        PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in TRAINING_SPLIT],
                         ecg_labels=[ecg_labels[index] for index in TRAINING_SPLIT], fs=fs,
                         augmentation_pipeline=AugmentationPipeline()),
        batch_size=args.batch_size, num_workers=args.batch_size, pin_memory=True,
        drop_last=False, shuffle=True)
    validation_dataset = DataLoader(
        PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in VALIDATION_SPLIT],
                         ecg_labels=[ecg_labels[index] for index in VALIDATION_SPLIT], fs=fs,
                         augmentation_pipeline=None),
        batch_size=args.batch_size, num_workers=args.batch_size, pin_memory=True,
        drop_last=False, shuffle=False)

    # Init model wrapper
    model_wrapper = ModelWrapper(network=network,
                                 optimizer=optimizer,
                                 loss_function=SoftmaxCrossEntropyLoss(weight=(0.2, 0.2, 1., 2.)),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 data_logger=data_logger,
                                 learning_rate_schedule=learning_rate_schedule,
                                 device=device)

    # Perform training
    model_wrapper.train(epochs=args.epochs)
