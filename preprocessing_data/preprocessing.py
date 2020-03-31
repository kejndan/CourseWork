import numpy as np
from scipy.stats import mode


class PreProcessing:
    def __init__(self, dataset):
        self.dataset = dataset

    def processing_missing_values(self, to='auto'):
        base_null = ['null', 'NULL', 'NaN', 'nan', '-', '?']
        np_dataset = np.array(self.dataset)
        if to == 'auto':
            index_no_del_features = []
            for feature in range(len(np_dataset[0])):
                index_missing_values = []
                index_filled_values = []
                for sample in range(len(np_dataset)):
                    if np_dataset[sample, feature] in base_null or np.isnan(np_dataset[sample, feature]):
                        index_missing_values.append(sample)
                    else:
                        index_filled_values.append(sample)
                if 1-len(index_missing_values)/len(np_dataset) > 0.2:
                    t = np_dataset[np.array(index_filled_values), feature]
                    mean = np_dataset[np.array(index_filled_values), feature].mean()
                    if len(np.array(index_missing_values)) != 0:
                        np_dataset[np.array(index_missing_values), feature] = mean
                    index_no_del_features.append(feature)
            np_dataset = np_dataset[:, index_no_del_features]
        elif to == 'mean' or to == 'median' or to == 'most_frequent':
            for feature in range(len(np_dataset[0])):
                index_missing_values = []
                index_filled_values = []
                for sample in range(len(np_dataset)):
                    if np_dataset[sample, feature] in base_null or np.isnan(np_dataset[sample, feature]):
                        index_missing_values.append(sample)
                    else:
                        index_filled_values.append(sample)
                if 1-len(index_missing_values)/len(np_dataset) > 0.2:
                    if to == 'mean':
                        value = np_dataset[np.array(index_filled_values), feature].mean()
                    elif to == 'median':
                        value = np.median(np_dataset[np.array(index_filled_values), feature])
                    elif to == 'most_frequent':
                        value = mode(np_dataset[np.array(index_filled_values), feature])[0][0]
                    if len(np.array(index_missing_values)) != 0:
                        np_dataset[np.array(index_missing_values), feature] = value
        else:
            np_dataset = np_dataset[np.logical_not(np.isnan(np_dataset))]
        return np_dataset


if __name__ == '__main__':
    a = np.array(np.random.rand(100)).reshape(10,10)
    a[1:3, 2] = np.nan
    a[3:5, 2] = 2
    pp = PreProcessing(a)
    b = pp.processing_missing_values(to='most_frequent')
    print(a.shape)
    print(b[1:3,2])


                    




