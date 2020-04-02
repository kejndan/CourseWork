import numpy as np
from scipy.stats import mode


class PreProcessing:
    def __init__(self, dataset):
        self.dataset = dataset
        self.np_dataset = np.array(self.dataset)

    def processing_missing_values(self, to='auto', features=None):
        base_null = ['null', 'NULL', 'NaN', 'nan', '-', '?']
        if features is None:
            features = range(len(self.np_dataset[0]))
        if to == 'auto':
            index_no_del_features = []
            for feature in features:
                index_missing_values = []
                index_filled_values = []
                for sample in range(len(self.np_dataset)):
                    if self.np_dataset[sample, feature] in base_null or np.isnan(self.np_dataset[sample, feature]):
                        index_missing_values.append(sample)
                    else:
                        index_filled_values.append(sample)
                if 1-len(index_missing_values)/len(self.np_dataset) > 0.2:
                    t = self.np_dataset[np.array(index_filled_values), feature]
                    mean = self.np_dataset[np.array(index_filled_values), feature].mean()
                    if len(np.array(index_missing_values)) != 0:
                        self.np_dataset[np.array(index_missing_values), feature] = mean
                    index_no_del_features.append(feature)
            self.np_dataset = self.np_dataset[:, index_no_del_features]
        elif to == 'mean' or to == 'median' or to == 'most_frequent':
            for feature in features:
                index_missing_values = []
                index_filled_values = []
                for sample in range(len(self.np_dataset)):
                    print(self.np_dataset[sample, feature])
                    if self.np_dataset[sample, feature] in base_null or np.isnan(self.np_dataset[sample, feature]):
                        index_missing_values.append(sample)
                    else:
                        index_filled_values.append(sample)
                if 1-len(index_missing_values)/len(self.np_dataset) > 0.2:
                    if to == 'mean':
                        value = self.np_dataset[np.array(index_filled_values), feature].mean()
                    elif to == 'median':
                        value = np.median(self.np_dataset[np.array(index_filled_values), feature])
                    elif to == 'most_frequent':
                        value = mode(self.np_dataset[np.array(index_filled_values), feature])[0][0]
                    if len(np.array(index_missing_values)) != 0:
                        self.np_dataset[np.array(index_missing_values), feature] = value
        else:
            self.np_dataset = self.np_dataset[np.logical_not(np.isnan(self.np_dataset))]
        return self.np_dataset

    def handling_outliners(self, method=None, factor=3, features=None):
        if features is None:
            features = range(len(self.np_dataset[0]))
        for feature in features:
            if method == 'std':
                upper_lim = self.np_dataset[:, feature].mean() + self.np_dataset[:, feature].std() * factor
                lower_lim = self.np_dataset[:, feature].mean() - self.np_dataset[:, feature].std() * factor
            else:
                upper_lim = np.quantile(self.np_dataset[:, feature], .95)
                lower_lim = np.quantile(self.np_dataset[:, feature], .05)

            if method == 'percentile' or method == 'std':
                self.np_dataset = self.np_dataset[(self.np_dataset[:, feature] < upper_lim) &
                                                  (self.np_dataset[:, feature] > lower_lim)]
            else:
                self.np_dataset[(self.np_dataset[:,feature] > upper_lim), feature] = upper_lim
                self.np_dataset[(self.np_dataset[:,feature] < lower_lim), feature] = lower_lim
        return self.np_dataset


if __name__ == '__main__':
    a = np.array(np.random.rand(100)).reshape(10,10)
    # a = np.array([[1,2,'nan'],[3,4,'nan'],['?',np.nan,3]])
    a[1:2, 2] = 1000
    a[3:5, 2] = -5
    pp = PreProcessing(a)
    b = pp.handling_outliners(factor=3, features=[2])
    print(np.round(a))
    print(np.round(b))


                    




