from collections import Counter
import numpy as np


def entropy_binning(x_data, y_data, n_bins, get_bins=True):
    """
    This function splits into bins on basis of entropy
    :param x_data: column to group into bins
    :param y_data: prediction column
    :param n_bins: number of bins
    :param get_bins: if true then return the segments bins, else then return column number
    :return: segments bins/column number
    """
    x_data, y_data = x_data.flatten(), y_data.flatten()
    merge_data = list(zip(x_data, y_data))  # merge x_data and y_data
    merge_data = sorted(merge_data, key=lambda x: x[0])  # sorted merge_data by x_data
    merge_data = np.array(merge_data).astype(np.float32)
    bins = np.zeros(n_bins - 1)  # create empty array for segments bins

    # 1st elem in concatenate is start of bins
    # 2nd elem in concatenate is result of entropy binning
    # 3rd elem in concatenate is end of bins
    bins = np.concatenate((np.array([merge_data[0][0]]),
                           __help_entropy_binning(merge_data, 0, n_bins - 1, bins),
                           np.array([merge_data[-1][0]])))

    if get_bins:
        return bins  # segments bins
    else:
        return np.digitize(x_data, bins, True)  # column number


def quantile_binning(x_data, n_bins, get_bins=True):
    """
    This function splits into bins basis of quantile
    :param x_data: column to group into bins
    :param n_bins: number of bins
    :param get_bins: if true then return the segments bins, else then return column number
    :return: segments bins/column number
    """
    x_data = x_data.flatten()
    step = 100 / n_bins / 100
    quantiles = np.quantile(x_data, np.arange(0, 1 + .0001, step))
    if get_bins:
        return quantiles
    else:
        return np.digitize(x_data, quantiles, True)


def equal_width_binning(x_data, n_bins, get_bins=True):
    """
    This function splits into equal bins basis of width(equal (x_data.max() - x_data.min())/n_bins)
    :param x_data: column to group into bins
    :param n_bins: number of bins
    :param get_bins: if true then return the segments bins, else then return column number
    :return: segments bins/column number
    """
    width = (x_data.max() - x_data.min()) / n_bins
    bins = np.arange(x_data.min(), x_data.max(), width)
    if get_bins:
        return bins
    else:
        return np.digitize(x_data, bins, True)


def custom_edge_binning(x_data, bins):
    """
    This function splits into bins basis of segments of user
    :param x_data: column to group into bins
    :param bins: segments of  user
    :return: column number
    """
    new_bins = np.concatenate((np.array([x_data.min()]),
                               bins,
                               np.array([x_data.max()])
                               ))
    return np.digitize(x_data, new_bins, True)


def custom_edge_equal_width_binning(x_data, start, width, end, get_bins=True):
    """
    This function splits into equal bins basis of start, width, end of user
    :param x_data: column to group into bins
    :param start: start of the first bin
    :param width: width of the bins
    :param end: end of the last bins
    :param get_bins: if true then return the segments bins, else then return column number
    :return: segments bins/column number
    """
    bins = np.arange(start, end, width)
    if get_bins:
        return bins
    else:
        return np.digitize(x_data, bins, True)


def __help_entropy_binning(merge_data, start, end, bins):
    """
    This function calculate bins with the minimum entropy
    :param merge_data: merge column for splits bins and prediction column
    :param start: index of start bins
    :param end: index of end bins
    :param bins: list of the segments bins
    :return: completed bins
    """
    if start >= end or merge_data is np.empty:
        return
    else:
        n_bins = end - start
        counter_data = Counter(merge_data[:, 1])
        counter_data = np.array(sorted(counter_data.items())).astype(np.float32)
        init_entropy = _entropy(counter_data[:, 1], sum(counter_data[:, 1]))
        best_entropy = init_entropy
        best_left_bin = counter_data
        best_right_bin = np.empty
        best_dif = merge_data[0, 0]
        for i in range(1, len(merge_data)):
            dif = (merge_data[i - 1][0] + merge_data[i][0]) / 2
            left_bin = merge_data[merge_data[:, 0] < dif]
            right_bin = merge_data[merge_data[:, 0] >= dif]
            counter_left_data, counter_right_data = Counter(left_bin[:, 1]), Counter(right_bin[:, 1])
            counter_left_data, counter_right_data = np.array(sorted(counter_left_data.items())).astype(np.int32), \
                                                    np.array(sorted(counter_right_data.items())).astype(np.int32)
            left_sum = sum(counter_left_data[:, 1])
            right_sum = sum(counter_right_data[:, 1])
            all_sum = left_sum + right_sum
            new_entropy = _entropy(counter_left_data[:, 1], left_sum) * left_sum / all_sum +\
                            _entropy(counter_right_data[:, 1], right_sum) * right_sum / all_sum
            if best_entropy > new_entropy:
                best_entropy = new_entropy
                best_dif = dif
                best_left_bin = left_bin
                best_right_bin = right_bin
        bins[start + n_bins // 2] = best_dif
        __help_entropy_binning(best_left_bin, start, n_bins // 2, bins), __help_entropy_binning(best_right_bin,
                                                                                                n_bins // 2 + 1, end,
                                                                                                bins)
        return bins


def _entropy(elements, sum_elements):
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


if __name__ == '__main__':
    print('This is module for binning')
