# coding=utf8
import numpy as np
from scipy.spatial.distance import pdist, squareform


def dist_corr(u, v):
    X = u[:, None]
    Y = v[:, None]
    n = len(X)
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    if np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)) == 0:
        return 0
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def entropy(elements, sum_elements):
    """
    This function calculate entropy for transmitted the values
    :param elements: values for calc entropy
    #TODO убрать sum_elements
    :param sum_elements:
    :return:
    """
    elem_entropy = 0
    for elem in elements:
        elem_entropy -= (elem / sum_elements) * np.log2(elem / sum_elements)
    return elem_entropy




