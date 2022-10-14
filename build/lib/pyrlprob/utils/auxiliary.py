import os
from typing import *
import numpy as np
import pandas as pd
import random
import gym
import tensorflow as tf
import torch
import collections.abc


def update(d, u):
    """
    Update the elements in dictionary d with those in dictionary u.
    """

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def set_global_seeds(seed: Optional[int]=None) -> List[int]:
    """
    Set the seed for python random, tensorflow, pytorch, numpy and gym spaces

    :param seed: (int) the seed
    """

    np.random.seed(seed)
    random.seed(seed)
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)

    return [seed]


def get_cp_dir_and_model(logdir: str, 
                         cp: int) -> Tuple[str, str]:
    """ 
    Get checkpoint directory and checkpoint model.

    Args:
        logdir (str): directory where the checkpoint is saved
        cp (int): checkpoint number
    
    Return:
        cp_dir (str): checkpoint directory
        cp_model (str): checkpoint model
    """

    cp_dir = logdir + "checkpoint_" + str(cp).zfill(6) + "/"
    cp_model = cp_dir + "checkpoint-" + str(cp)

    assert (os.path.exists(cp_dir)), "Folder %s does not exist!." % (cp_dir)

    return cp_dir, cp_model


def moving_average(values: np.ndarray,
                   window: int) -> np.ndarray:
    """
    Smooth values by doing a moving average

    Args:
        values (numpy array): actual values
        window (int): window's width
    
    Return:
        smoothed_values (numpy array): smoothed values
    """

    weights = np.repeat(1.0, window) / window
    smoothed_values = np.convolve(values, weights, 'valid')

    return smoothed_values


def column_progress(filename: str,
                    column: str, 
                    last_row: Optional[int] = None) -> List[Any]:
    """
    Return the list of values in a given column of a .csv file

    Args:
        filename (str): name of the .csv file with all the data
        column (str): header of the column of interest
        last_row (int): last raw of the column to process
    
    Return:
        column_data (list): list of values in the column
    """

    data = pd.read_csv(filename) 
    head = data.head(last_row)
    column_data = list(head[column])

    return column_data


def metric_training_trend(metric_name: str,
                          experiment_dirs: List[str],
                          last_checkpoints: List[int]) -> List[Any]:
    """
    Retrieve the trend of a metric during training

    Args:
        metric_name (str): full name of the metric (with the path)
        experiment_dirs (list): list of the experiment directories
        last_checkpoints (list): list with the last checkpoints of the experiments
    
    Return:
        metric_trend (list): list with the values of the metric
    """

    metric_trend = []
    for exp_num, exp_dir in enumerate(experiment_dirs):
        metric_trend = metric_trend + column_progress(exp_dir+"progress.csv", \
            metric_name, last_checkpoints[exp_num])
    
    return metric_trend