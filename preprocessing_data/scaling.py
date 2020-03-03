import numpy as np


def normalization(data):
    norm_data = (data - data.min())/(data.max() - data.min())
    return norm_data


def standardization(data):
    stand_data = (data - data.mean())/(data.std())
    return stand_data


def l2_normalized(data):
    l2_norm_data = data/np.linalg.norm(data)
    return l2_norm_data

