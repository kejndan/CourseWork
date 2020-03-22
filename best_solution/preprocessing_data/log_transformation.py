import numpy as np
from scipy import stats


def to_log(data, arg=10, only_positive=False):
    if only_positive:
        return np.log10(data)/np.log10(arg)
    else:
        return np.log10(data+1)/np.log10(arg)


def to_box_cox(data, lmbd=None):
    data = np.where(data > 0, data.copy(), .0001)
    return stats.boxcox(data, lmbd)