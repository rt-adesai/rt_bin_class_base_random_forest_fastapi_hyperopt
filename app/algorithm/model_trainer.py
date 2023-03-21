#!/usr/bin/env python

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pprint

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
import numpy as np
import pandas as pd
from algorithm.model.classifier import Classifier
from algorithm.utils import get_model_config
from sklearn.utils import shuffle

# get model configuration parameters
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):

    # set random seeds
    utils.set_seeds()

    # perform train/valid split
    train_data = data
    # print('train_data shape:',  train_data.shape)

    # preprocess data
    print("Pre-processing data...")
    train_data, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)
    train_X, train_y = train_data["X"].astype(np.float), train_data["y"].astype(
        np.float
    )
    # print('train_X/y shape:',  train_X.shape, train_y.shape)

    # balance the target classes
    train_X, train_y = get_resampled_data(train_X, train_y)

    # Create and train model
    print("Fitting model ...")
    model = train_model(train_X, train_y, hyper_params)

    return preprocess_pipe, model


def train_model(train_X, train_y, hyper_params):
    # get model hyper-paameters parameters
    # data_based_params = get_data_based_model_params(train_X)
    # Create and train model
    model = Classifier(**hyper_params)
    # model.summary()
    model.fit(train_X, train_y)
    # print("last_loss:", history.history['loss'][-1])
    return model


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg)
    # pprint.pprint(pp_params)
    # sys.exit()

    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)

    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
    # print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
    return train_data, valid_data, preprocess_pipe


def get_resampled_data(X, y):

    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # resamples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time).
    max_resample = model_cfg["max_resample_of_minority_classes"]
    unique, class_count = np.unique(y, return_counts=True)
    # class_count = [ int(c) for c in class_count]
    max_obs_count = max(class_count)

    resampled_X, resampled_y = [], []
    for i, count in enumerate(class_count):
        if count == 0:
            continue
        # find total num_samples to use for this class
        size = (
            max_obs_count
            if max_obs_count / count < max_resample
            else count * max_resample
        )
        # if observed class is 50 samples, and we need 125 samples for this class,
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples

        full_samples = size // count
        idx = y == i
        for _ in range(full_samples):
            resampled_X.append(X[idx, :])
            resampled_y.append(y[idx])
        # find the remaining samples to draw randomly
        remaining = size - count * full_samples
        sampled_idx = np.random.randint(count, size=remaining)
        resampled_X.append(X[idx, :][sampled_idx, :])
        resampled_y.append(y[idx][sampled_idx])

    resampled_X = np.concatenate(resampled_X, axis=0)
    resampled_y = np.concatenate(resampled_y, axis=0)
    # print(resampled_X.shape, resampled_y.shape)

    # shuffle the arrays
    resampled_X, resampled_y = shuffle(resampled_X, resampled_y)
    return resampled_X, resampled_y
