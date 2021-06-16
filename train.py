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
                             "ECGAttNet_L", "ECGAttNet_XL", "ECGAttNet_XXL", "ECGAttNet_130M"],
                    help="Type of network configuration to be utilized.")
parser.add_argument("--load_network", default=None, type=str,
                    help="If set given network (state dict) is loaded.")
parser.add_argument("--no_signal_encoder", default=False, action="store_true",
                    help="Binary flag. If set no signal encoder is utilized.")
parser.add_argument("--no_spectrogram_encoder", default=False, action="store_true",
                    help="Binary flag. If set no spectrogram encoder is utilized.")
parser.add_argument("--icentia11k", default=False, action="store_true",
                    help="Binary flag. If set icentia11k dataset is utilized.")
parser.add_argument("--challange", default=False, action="store_true",
                    help="Binary flag. If set challange split is utilized.")
parser.add_argument("--two_classes", default=False, action="store_true",
                    help="Binary flag. If set two classes are utilized. "
                         "Can only used with PhysioNet dataset and challange flag.")

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
    if args.physio_net:
        dataset_info = "_physio_net_dataset"
    elif args.icentia11k:
        dataset_info = "_icentia11k_dataset"
    else:
        dataset_info = "_default_dataset"
    if args.challange:
        dataset_info += "_challange"
    if args.two_classes:
        dataset_info += "_two_classes"
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
    elif args.network_config == "ECGAttNet_XXL":
        config = ECGAttNet_CONFIG_XXL
        data_logger = Logger(experiment_path_extension="ECGAttNet_XXL" + dataset_info)
        print("ECGAttNet_XXL utilized")
    else:
        config = ECGAttNet_CONFIG_130M
        data_logger = Logger(experiment_path_extension="ECGAttNet_130M" + dataset_info)
        print("ECGAttNet_130M utilized")

    # Not dropout of no data augmentation
    if args.no_data_aug:
        config["dropout"] = 0.

    # Check flags and set classes
    if args.two_classes:
        assert (not args.icentia11k) and args.challange and args.physio_net, \
            "Two class flag can only be used if incentia11k flag is not set and challange as well as " \
            "pyhsio_net flags are set"
        config["classes"] = 2
        config["dropout"] = 0.3
        augmentation_pipeline_config = AUGMENTATION_PIPELINE_CONFIG_2D
    else:
        augmentation_pipeline_config = AUGMENTATION_PIPELINE_CONFIG

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
        state_dict = torch.load(args.load_network)
        model_state_dict = network.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if model_state_dict[key].shape == value.shape}
        model_state_dict.update(state_dict)
        network.load_state_dict(model_state_dict)
        print("Network loaded")

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
            batch_size=max(1, args.batch_size // 50), num_workers=min(max(args.batch_size // 50, 4), 20),
            pin_memory=True, drop_last=False, shuffle=True, collate_fn=icentia11k_dataset_collate_fn)
        validation_dataset = DataLoader(
            Icentia11kDataset(path=args.dataset_path, split=VALIDATION_SPLIT_ICENTIA11K,
                              random_seed=VALIDATION_SEED_ICENTIA11K),
            batch_size=max(1, args.batch_size // 50), num_workers=min(max(args.batch_size // 50, 4), 20),
            pin_memory=True, drop_last=False, shuffle=False, collate_fn=icentia11k_dataset_collate_fn)
    else:
        ecg_leads, ecg_labels, fs, ecg_names = load_references(args.dataset_path)
        if args.physio_net:
            if args.challange:
                print("Challange split is utilized")
                if args.two_classes:
                    training_split = TRAINING_SPLIT_CHALLANGE_2_CLASSES
                    validation_split = VALIDATION_SPLIT_CHALLANGE_2_CLASSES
                else:
                    training_split = TRAINING_SPLIT_CHALLANGE
                    validation_split = VALIDATION_SPLIT_CHALLANGE
            else:
                training_split = TRAINING_SPLIT_PHYSIONET
                validation_split = VALIDATION_SPLIT_PHYSIONET
        else:
            training_split = TRAINING_SPLIT
            validation_split = VALIDATION_SPLIT
        training_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in training_split],
                             ecg_labels=[ecg_labels[index] for index in training_split], fs=fs,
                             augmentation_pipeline=None if args.no_data_aug else AugmentationPipeline(
                                 AUGMENTATION_PIPELINE_CONFIG),
                             two_classes=args.two_classes),
            batch_size=args.batch_size, num_workers=min(args.batch_size, 20), pin_memory=True,
            drop_last=False, shuffle=True)
        validation_dataset = DataLoader(
            PhysioNetDataset(ecg_leads=[ecg_leads[index] for index in validation_split],
                             ecg_labels=[ecg_labels[index] for index in validation_split], fs=fs,
                             augmentation_pipeline=None,
                             two_classes=args.two_classes),
            batch_size=args.batch_size, num_workers=min(args.batch_size, 20), pin_memory=True,
            drop_last=False, shuffle=False)

    # Make loss weights
    if args.icentia11k:
        weights = (1., 1., 1., 1., 1., 1., 1.)
    elif args.two_classes:
        weights = (1., 1.)
    else:
        weights = (0.4, 0.7, 0.9, 0.9)

    # Init model wrapper
    model_wrapper = ModelWrapper(network=network,
                                 optimizer=optimizer,
                                 loss_function=SoftmaxCrossEntropyLoss(weight=weights),
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 data_logger=data_logger,
                                 learning_rate_schedule=learning_rate_schedule,
                                 device=device)

    # Perform training
    model_wrapper.train(epochs=args.epochs)
