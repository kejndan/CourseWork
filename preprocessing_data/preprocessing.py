import numpy as np
from scipy.stats import mode
from preprocessing_data import binning
import pandas as pd
from preprocessing_data.log_transformation import to_log, to_box_cox
from preprocessing_data import scaling


class PreProcessing:
    def __init__(self, dataset, index_target=None):
        self.dataset = pd.DataFrame(dataset)
        if index_target is None:
            index_target = -1
        self.target = np.array(self.dataset[self.dataset.columns[index_target]], dtype=np.object)
        self.np_dataset = np.array(self.dataset.drop(self.dataset.columns[index_target], 1), dtype=np.object)
        self.one_hot_features = []

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
                    if self.np_dataset[sample, feature] in base_null or type(self.np_dataset[sample, feature]) == str \
                            or np.isnan(self.np_dataset[sample, feature]):
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
            full_dataset = np.concatenate((self.np_dataset, self.target[:,None]), axis=1)
            full_dataset_without_nan = np.array(pd.DataFrame(full_dataset).dropna())
            self.np_dataset = full_dataset_without_nan[:,:-1]
            self.target = full_dataset_without_nan[:,-1:]
        return self.np_dataset

    def handling_outliners(self, method=None, factor=3, features=None):
        # TODO mean and median
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

    def binning(self, n_bins, type_binning='equal', features=None):
        if features is None:
            features = range(len(self.np_dataset[0]))
        for feature in features:
            if type_binning == 'equal':
                self.np_dataset[:, feature] = binning.equal_width_binning(self.np_dataset[:, feature], n_bins, False)\
                    .astype(str)
            elif type_binning == 'entropy':
                self.np_dataset[:, feature] = binning.entropy_binning(self.np_dataset[:, feature], self.target, n_bins,
                                                                      False).astype(str)
            else:
                self.np_dataset[:, feature] = binning.quantile_binning(self.np_dataset[:, feature], n_bins, False)\
                    .astype(str)

            return self.np_dataset

    def transform(self, type_transform='log', arg=10, features=None):
        if features is None:
            features = range(len(self.np_dataset[0]))
        for feature in features:
            if type(self.np_dataset[0,feature]) != str:
                print(self.np_dataset[:, feature])
                if type_transform == 'log':
                    self.np_dataset[:, feature] = to_log(self.np_dataset[:, feature], arg)
                elif type_transform == 'box-cox':
                    self.np_dataset[:, feature] = to_box_cox(self.np_dataset[:, feature])
            return self.np_dataset

    def scaling(self, type_scale='norm', features=None):
        if features is None:
            features = range(len(self.np_dataset[0]))
        for feature in features:
            if type(self.np_dataset[0, feature]) != str :
                if type_scale == 'norm':
                    self.np_dataset[:, feature] = scaling.normalization(self.np_dataset[:, feature])
                elif type_scale == 'stand':
                    self.np_dataset[:, feature] = scaling.standardization(self.np_dataset[:, feature])
                elif type_scale == 'l2-norm':
                    self.np_dataset[:, feature] = scaling.l2_normalized(self.np_dataset[:, feature])
            return self.np_dataset

    def preprocessing_manager(self, kwargs):
        if 'processing_missing_values' in kwargs:
            if 'to' not in kwargs['processing_missing_values']:
                kwargs['processing_missing_values']['to'] = 'auto'
            if 'features' not in kwargs['processing_missing_values']:
                kwargs['processing_missing_values']['features'] = None
            self.processing_missing_values(kwargs['processing_missing_values']['to'],
                                           kwargs['processing_missing_values']['features'])
        if 'handling_outliners' in kwargs:
            if 'method' not in kwargs['handling_outliners']:
                kwargs['handling_outliners']['method'] = None
            if 'factor' not in kwargs['handling_outliners']:
                kwargs['handling_outliners']['factor'] = 3
            if 'features' not in kwargs['handling_outliners']:
                kwargs['handling_outliners']['features'] = None
            self.handling_outliners(kwargs['handling_outliners']['method'],kwargs['handling_outliners']['factor'],
                                    kwargs['handling_outliners']['features'])
        if 'binning' in kwargs:
            if 'n_bins' not in kwargs['binning']:
                kwargs['binning']['n_bins'] = 3
            if 'type_binning' not in kwargs['binning']:
                kwargs['binning']['type_binning'] = 'equal'
            if 'features' not in kwargs['binning']:
                kwargs['binning']['features'] = None
            self.binning(kwargs['binning']['n_bins'],kwargs['binning']['type_binning'],kwargs['binning']['features'])
        if 'transform' in kwargs:
            if 'type_transform' not in kwargs['transform']:
                kwargs['transform']['type_transform'] = 'log'
            if 'arg' not in kwargs['transform']:
                kwargs['transform']['arg'] = 10
            if 'features' not in kwargs['transform']:
                kwargs['transform']['features'] = None
            self.transform(kwargs['transform']['type_transform'], kwargs['transform']['arg'],
                           kwargs['transform']['features'])
        if 'scaling' in kwargs:
            if 'type_scale' not in kwargs['scaling']:
                kwargs['scaling']['type_scale'] = 'norm'
            if 'features' not in kwargs['scaling']:
                kwargs['scaling']['features'] = None
            self.scaling(kwargs['scaling']['type_scale'], kwargs['scaling']['features'])
        self.one_hot_features = []
        for index in range(len(self.np_dataset[0,:])):
            if type(self.np_dataset[:,index]) == str:
                self.one_hot_features.append(index)
        return self.np_dataset







if __name__ == '__main__':
    # a = np.random.rand(100).reshape(10, 10)*100
    # b = np.random.randint(0, 2, (10,1))
    # dataset = np.concatenate((a,b),axis=1)
    # print(dataset)

    # q = pp.binning(3, features=[0])
    # print(q)
    # df = pd.read_csv('../datasets/housing.csv')
    a = np.array([[1,2,'o',1],[3,4,'p',0],[np.nan,0,'q',0],[4,100,'f',1]],dtype=np.object)
    pp = PreProcessing(a, -1)
    print(pp.np_dataset)
    rules = {'processing_missing_values':{'to':'del'}, 'handling_outliners':{'method':None, 'features':[1]},
             'binning':{'n_bins':3, 'features':[1]},'transform':{}}
    pp.preprocessing_manager(rules)
    print(pp.np_dataset)
    print(pp.target)

                    




