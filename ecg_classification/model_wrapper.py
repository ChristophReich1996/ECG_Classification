from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from rtpt.rtpt import RTPT

from .logger import Logger
from .metric import Accuracy, F1


class ModelWrapper(object):
    """
    This class implements a wrapper for the training and validation loop.
    """

    def __init__(self,
                 network: Union[nn.Module, nn.DataParallel],
                 optimizer: Optimizer,
                 loss_function: nn.Module,
                 training_dataset: DataLoader,
                 validation_dataset: DataLoader,
                 data_logger: Logger,
                 learning_rate_schedule: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
                 device: str = "cuda") -> None:
        """
        Constructor method
        :param network: (Union[nn.Module, nn.DataParallel]) Network to be trained
        :param optimizer: (Optimizer) Optimizer module
        :param loss_function: (nn.Module) Loss function
        :param training_dataset: (DataLoader) Training dataset
        :param validation_dataset: (DataLoader) Validation dataset
        :param data_logger: (Logger) Data logger
        :param learning_rate_schedule: (Optional[torch.optim.lr_scheduler.MultiStepLR]=NoneR) Learning rate schedule
        :param device: (str) Device to be utilize
        """
        # Save parameters
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.data_logger = data_logger
        self.learning_rate_schedule = learning_rate_schedule
        self.device = device

    def train(self, epochs: int = 100, validate_after_n_epochs: int = 10, save_model_after_n_epochs: int = 10,
              save_best_model: int = True) -> None:
        """
        Training method
        :param epochs: (Optional[int]) Number of epochs to be performed
        :param validate_after_n_epochs: (Optional[int]) Number of epochs after the model gets evaluated
        :param save_model_after_n_epochs: (Optional[int]) Number of epochs after the model is saved
        :param save_best_model: (Optional[bool]) If true the best model performing model based on validation is saved
        """
        # Network into train mode
        self.network.train()
        # Model to device
        self.network.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset))
        # Init best validation metric
        best_validation_metric = 0.0
        # Init rtpt
        rtpt = RTPT(name_initials="CR", experiment_name="ECG_Class.", max_iterations=epochs)
        # Start rtpt
        rtpt.start()
        # Main training loop
        for self.epoch in range(epochs):
            # Rtpt step
            rtpt.step(subtitle="ACC={:.3f}".format(best_validation_metric))
            for batch in self.training_dataset:
                # Update progress bar
                self.progress_bar.update(n=1)
                # Reset gradients of model
                self.optimizer.zero_grad()
                # Unpack batch
                ecg_leads, spectrogram, labels = batch
                # Data to device
                ecg_leads = ecg_leads.to(self.device)
                spectrogram = spectrogram.to(self.device)
                labels = labels.to(self.device)
                # Make prediction
                predictions = self.network(ecg_leads, spectrogram)
                # Calc loss
                loss = self.loss_function(predictions, labels)
                # Calc backward pass
                loss.backward()
                # Perform optimization step
                self.optimizer.step()
                # Show loss in progress bar
                self.progress_bar.set_description(
                    "Epoch {}/{} L={:.4f} ACC={:.4f}".format(self.epoch + 1, epochs, loss.item(),
                                                             best_validation_metric))
                # Log loss
                self.data_logger.log_metric(metric_name="Loss", value=loss.cpu().item())
            # Update learning rate schedule
            if self.learning_rate_schedule is not None:
                self.learning_rate_schedule.step()
            # Perform validation
            if (self.epoch + 1) // validate_after_n_epochs:
                current_validation_metric = self.validate()
                # Check if best model
                if current_validation_metric > best_validation_metric:
                    best_validation_metric = current_validation_metric
                    if save_best_model:
                        self.data_logger.save_model(model_sate_dict=self.network.state_dict(), name="best_model")
            # Save model
            if (self.epoch + 1) // save_model_after_n_epochs:
                self.data_logger.save_model(model_sate_dict=self.network.state_dict(), name=str(self.epoch + 1))
            # Save logs
            self.data_logger.save()
        # Final validation
        current_validation_metric = self.validate()
        # Check if best model
        if current_validation_metric > best_validation_metric:
            if save_best_model:
                self.data_logger.save_model(model_sate_dict=self.network.state_dict(), name="best_model")

    @torch.no_grad()
    def validate(self, validation_metrics: Tuple[nn.Module, ...] = (F1(), Accuracy())) -> float:
        """
        Validation method
        :param validation_metrics: (Tuple[nn.Module, ...]) Tuple of validation metrics (last one is returned)
        :return: (float) Validation metric (last metric in tuple)
        """
        # Network into eval mode
        self.network.eval()
        # Network to device
        self.network.to(self.device)
        # Show validation in progress bar
        try:
            self.progress_bar.set_description("Validation")
        except AttributeError as e:
            pass
        # Validation loop
        for batch in self.validation_dataset:
            # Unpack batch
            ecg_leads, spectrogram, labels = batch
            # Data to device
            ecg_leads = ecg_leads.to(self.device)
            spectrogram = spectrogram.to(self.device)
            labels = labels.to(self.device)
            # Make prediction
            predictions = self.network(ecg_leads, spectrogram)
            # Calc loss
            loss = self.loss_function(predictions, labels)
            # Track loss
            self.data_logger.log_temp_metric(metric_name="validation_loss", value=loss.item())
            # Compute all validation metrics
            for validation_metric in validation_metrics:
                self.data_logger.log_temp_metric(metric_name=str(validation_metric),
                                                 value=validation_metric(predictions, labels))
        # Average metrics
        metric_results = self.data_logger.save_temp_metric(
            metric_name=["validation_loss"] + [str(validation_metric) for validation_metric in validation_metrics])
        # Model back into training mode
        self.network.train()
        return metric_results[str(validation_metrics[-1])]
