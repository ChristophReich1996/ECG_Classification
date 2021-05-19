from argparse import ArgumentParser
import os

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--cuda_devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--no_data_aug", default=False, action="store_true",
                    help="Binary flag. If set no data augmentation is utilized.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If set data parallel is utilized.")
parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--lr", default=1e-03, type=float,
                    help="Learning rate to be employed.")
parser.add_argument("--batch_size", default=24, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--physio_net", default=False, action="store_true",
                    help="Binary flag. Utilized PhysioNet dataset instead of default one.")
parser.add_argument("--dataset_path", default="data/training/", type=str,
                    help="Path to dataset.")
parser.add_argument("--network_config", default="ECGCNN_M", type=str,
                    choices=["ECGCNN_S", "ECGCNN_M", "ECGCNN_L", "ECGCNN_XL", "ECGAttNet_S", "ECGAttNet_M",
                             "ECGAttNet_L", "ECGAttNet_XL", "ECGAttNet_XXL"],
                    help="Type of network configuration to be utilized.")
parser.add_argument("--load_network", default=None, type=str,
                    help="If set given network (state dict) is loaded.")
parser.add_argument("--no_signal_encoder", default=False, action="store_true",
                    help="Binary flag. If set no signal encoder is utilized.")
parser.add_argument("--no_spectrogram_encoder", default=False, action="store_true",
                    help="Binary flag. If set no spectrogram encoder is utilized.")
parser.add_argument("--icentia11k", default=False, action="store_true",
                    help="Binary flag. If set icentia11k dataset is utilized.")
# Get arguments
args = parser.parse_args()

# Set device type
device = "cuda"

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import torch
import torch_optimizer
from torch.utils.data import DataLoader

from wettbewerb import load_references
from ecg_classification import *

if __name__ == '__main__':
    # Add dataset info
    dataset_info = "_default_dataset" if not args.physio_net else "_physio_net_dataset"
    # Init network
    if args.network_config == "ECGCNN_S":
        config = ECGCNN_CONFIG_S
        data_logger = Logger(experiment_path_extension="ECGCNN_S" + dataset_info)
        print("ECGCNN_S utilized")
    elif args.network_config == "ECGCNN_M":
        config = ECGCNN_CONFIG_M
        data_logger = Logger(experiment_path_extension="ECGCNN_M" + dataset_info)
        print("ECGCNN_M utilized")
    elif args.network_config == "ECGCNN_L":
        config = ECGCNN_CONFIG_L
        data_logger = Logger(experiment_path_extension="ECGCNN_L" + dataset_info)
        print("ECGCNN_L utilized")
    elif args.network_config == "ECGCNN_XL":
        config = ECGCNN_CONFIG_XL
        data_logger = Logger(experiment_path_extension="ECGCNN_XL" + dataset_info)
        print("ECGCNN_XL utilized")
    elif args.network_config == "ECGAttNet_S":
        config = ECGAttNet_CONFIG_S
        data_logger = Logger(experiment_path_extension="ECGAttNet_S" + dataset_info)
        print("ECGAttNet_S utilized")
    elif args.network_config == "ECGAttNet_M":
        config = ECGAttNet_CONFIG_M
        data_logger = Logger(experiment_path_extension="ECGAttNet_M" + dataset_info)
        print("ECGAttNet_M utilized")
    elif args.network_config == "ECGAttNet_L":
        config = ECGAttNet_CONFIG_L
        data_logger = Logger(experiment_path_extension="ECGAttNet_L" + dataset_info)
        print("ECGAttNet_L utilized")
    elif args.network_config == "ECGAttNet_XL":
        config = ECGAttNet_CONFIG_XL
        data_logger = Logger(experiment_path_extension="ECGAttNet_XL" + dataset_info)
        print("ECGAttNet_XL utilized")
    else:
        config = ECGAttNet_CONFIG_XXL
        data_logger = Logger(experiment_path_extension="ECGAttNet_XXL" + dataset_info)
        print("ECGAttNet_XXL utilized")

    # Not dropout of no data augmentation
    if args.no_data_aug:
        config["dropout"] = 0.

    # Change number of classes if icentia11k dataset is used
    if args.icentia11k:
        config["classes"] = 7

    if "CNN" in args.network_config:
        network = ECGCNN(config=config)
    else:
        network = ECGAttNet(config=config)

    # Set used of encoders for ablation
    if args.no_signal_encoder:
        network.no_signal_encoder = True
        print("No signal encoder is utilized")
    if args.no_spectrogram_encoder:
        network.no_spectrogram_encoder = True
        print("No spectrogram encoder is utilized")

    # Load network
    if args.load_network is not None:
        network.load_state_dict(torch.load(args.load_network))

    # Print network parameters
    print("# parameters:", sum([p.numel() for p in network.parameters()]))

    # Init data parallel if utilized
    network = torch.nn.DataParallel(network)

    # Init optimizer
    optimizer = torch_optimizer.RAdam(params=network.parameters(), lr=args.lr)

    # Init learning rate schedule
    learning_rate_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[1 * args.epochs // 4, 2 * args.epochs // 4, 3 * args.epochs // 4], gamma=0.1)

    # Init datasets
    if args.icentia11k:
        training_dataset = DataLoader(
            Icentia11kDataset(path=args.dataset_path, split=TRAINING_SPLIT_ICENTIA11K),
            batch_size=max(1, args.batch_size // 50), num_workers=min(args.batch_size // 50, 20), pin_memory=True,
            drop_last=False, shuffle=True, collate_fn=icentia11k_dataset_collate_fn)
        validation_dataset = DataLoader(
            Icentia11kDataset(path=args.dataset_path, split=VALIDATION_SPLIT_ICENTIA11K,
                              random_seed=VALIDATION_SEED_ICENTIA11K),
            batch_size=max(1, args.batch_size // 50), num_workers=min(args.batch_size // 50, 20), pin_memory=True,
            drop_last=False, shuffle=False, collate_fn=icentia11k_dataset_collate_fn)
    else:
        ecg_leads, ecg_labels, fs, ecg_names = load_references(args.dataset_path)
        training_split = TRAINING_SPLIT if not args.physio_net else TRAINING_SPLIT_PHYSIONET
        validation_split = VALIDATION_SPLIT if not args.physio_net else VALIDATION_SPLIT_PHYSIONET
        training_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in training_split],
                             ecg_labels=[ecg_labels[index] for index in training_split], fs=fs,
                             augmentation_pipeline=None if args.no_data_aug else AugmentationPipeline(
                                 AUGMENTATION_PIPELINE_CONFIG)),
            batch_size=args.batch_size, num_workers=min(args.batch_size, 20), pin_memory=True,
            drop_last=False, shuffle=True)
        validation_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in validation_split],
                             ecg_labels=[ecg_labels[index] for index in validation_split], fs=fs,
                             augmentation_pipeline=None),
            batch_size=args.batch_size, num_workers=min(args.batch_size, 20), pin_memory=True,
            drop_last=False, shuffle=False)

    # Init model wrapper
    model_wrapper = ModelWrapper(network=network,
                                 optimizer=optimizer,
                                 loss_function=SoftmaxCrossEntropyLoss(
                                     weight=(0.4, 0.7, 0.9, 0.9)
                                     if not args.icentia11k else (1., 1., 1., 1., 1., 1., 1.)),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 data_logger=data_logger,
                                 learning_rate_schedule=learning_rate_schedule,
                                 device=device)

    # Perform training
    model_wrapper.train(epochs=args.epochs)
