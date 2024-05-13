from create_model.parameters import Params
from create_model.model import Make_model

import os
import time
import datetime
import numpy as np
from typing import Union, Tuple

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup


# store best statistics
class Best_stats():
    """
    Class to store the accuracy, loss, and f1 scores of the best model
    during training. In addition, provides a method to save the data of
    the current best model.
    """

    def __init__(
        self,
        best_averaged_train_accuracy = 0,
        best_averaged_train_loss = 1e6,
        best_averaged_train_f1_score = 0,
        best_averaged_val_accuracy = 0,
        best_averaged_val_loss = 1e6,
        best_averaged_val_f1_score = 0
    ) -> None:
        """Initialize instance of Best_stats with current best of model.

        Keyword arguments:
        best_averaged_train_accuracy -- training accuracy of best model (default 0)
        best_averaged_train_loss -- training loss of best model (default 1e6)
        best_averaged_train_f1_score -- training f1 score of best model (default 0)
        best_averaged_val_accuracy -- validation accuracy of best model (default 0)
        best_averaged_val_loss -- validation loss of best model (default 1e6)
        best_averaged_val_f1_score -- validation f1 score of best model (default 0)
        """

        self.best_average_train_accuracy = best_averaged_train_accuracy
        self.best_average_train_loss = best_averaged_train_loss
        self.best_average_train_f1_score = best_averaged_train_f1_score
        self.best_average_val_accuracy = best_averaged_val_accuracy
        self.best_average_val_loss = best_averaged_val_loss
        self.best_average_val_f1_score = best_averaged_val_f1_score
        
    def __str__(self) -> str:
        """
        Returns attribute names and current values corresponding to best model.
        """
        
        # get attributes of class
        attrs = vars(self)
        
        return "\n".join("%s: %s" % item for item in attrs.items())
        
    def record_best(
        self,
        model: Make_model,
        average_train_accuracy: float,
        average_train_loss: float,
        average_train_f1_score: float,
        average_val_accuracy: float,
        average_val_loss: float,
        average_val_f1_score: float
    ) -> None:
        """
        If validation f1 score exceeds current best then update
        all information with the current model's scores and save
        model state.

        Keyword arguments:
        model -- current model in training
        average_train_accuracy -- training accuracy of model
        average_train_loss -- training loss of model
        average_train_f1_score -- training f1 score of model
        average_val_accuracy -- validation accuracy of model
        average_val_loss -- validation loss of model
        average_val_f1_score -- validation f1 score of model
        """
        
        # track best performance and save the model's state
        if average_val_f1_score > self.best_average_val_f1_score:

            # update best model scores
            self.best_average_train_accuracy = average_train_accuracy
            self.best_average_train_loss = average_train_loss
            self.best_average_train_f1_score = average_train_f1_score
            self.best_average_val_accuracy = average_val_accuracy
            self.best_average_val_loss = average_val_loss
            self.best_average_val_f1_score = average_val_f1_score

            # check if file for data exists and create if does not
            os.makedirs("model", exist_ok=True)

            # save path
            model_path = os.path.join("model", "model.pth")

            # save model
            torch.save(model.model.state_dict(), model_path)