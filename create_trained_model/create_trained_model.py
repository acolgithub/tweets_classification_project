from create_model.parameters import Params
from create_model.preprocessing import preprocess_text
from create_model.model import Make_model
from create_model.save_best import Best_stats

import os

import time
import datetime
import re
import contractions
import pandas as pd
import numpy as np
from typing import Union, Tuple

# natural language processing
import nltk

# metric
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler

# using version 4.20.1
from transformers import DebertaTokenizer
from transformers import logging
from transformers import DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup


def create_trained_model() -> None:
    """Trains a DeBERTa model from loaded in training data."""

    # try and read in data
    try:
        # read in data
        train_data = pd.read_csv("./create_model/data/train.csv")
        test_data = pd.read_csv("./create_model/data/test.csv")

    # if data is not found then request user to obtain it
    except FileNotFoundError:
        print("""
                File not found:
                Please ensure the current directory contains a folder called "data".
                Then add the training set (train.csv) and the testing set (test.csv) to "data".
            """)


    # check if cuda is available else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set parameters
    params = Params(device = device)

    # split training data into text and targets
    text = train_data.text.values.tolist()
    y = torch.tensor(train_data.target)

    # preprocess text
    X = preprocess_text(text, params)

    # create an instance of deberta model
    test_model = Make_model(params)

    # initialize current best model stats
    model_best_stats = Best_stats()

    # add adamW optimizer
    optimizer = AdamW(
        params=test_model.model.parameters(),
        lr=params.lr,
        betas=params.betas,
        eps=params.eps
    )

    # form dataset
    dataset = list(zip(X, y.reshape(y.shape[0], 1)))

    # set sizes
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # split data
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # make training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = params.batch_size
    )

    # make testing dataloader
    test_dataloader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size = params.batch_size
    )

    # get linear learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = len(train_dataloader) * params.n_epochs
    )

    # train model over strata
    for epoch in range(params.n_epochs):

        # print epoch
        print(f"Epoch: {epoch + 1}/{params.n_epochs}")

        # train model
        average_train_accuracy, average_train_loss, average_train_f1_score = test_model.train(train_dataloader, optimizer, scheduler, params)

        # test model
        average_val_accuracy, average_val_loss, average_val_f1_score = test_model.test(test_dataloader, params)

        # record best model
        model_best_stats.record_best(
            test_model,
            average_train_accuracy,
            average_train_loss,
            average_train_f1_score,
            average_val_accuracy,
            average_val_loss,
            average_val_f1_score
        )

        print(model_best_stats)
        print("\n")



if __name__ == "__main__":
    create_trained_model()

