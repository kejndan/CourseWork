# coding=utf8
import numpy as np
from scipy.stats import mode
from preprocessing_data import binning
import pandas as pd
from preprocessing_data.log_transformation import to_log, to_box_cox
from preprocessing_data import scaling
from sklearn.preprocessing import OneHotEncoder


class PreProcessing:
    """
    Данный класс используется для предобработки датасета:
    *обработка пропущенных значений
    *обработка выбросов
    *категоризация(binning) признака
    *трансформация признака
    *уменьшений значений(scaling)
    """

    def __init__(self, dataset, index_target=None) :
        """
        :param dataset: полный датасет(X и Y)
        :param index_target: индекс столбца Y
        """
        self.dataset = pd.DataFrame(dataset)
        if index_target is None :
            index_target = -1
        self.target = np.array(self.dataset[self.dataset.columns[index_target]], dtype=np.object)
        self.np_dataset = np.array(self.dataset.drop(self.dataset.columns[index_target], 1), dtype=np.object)
        self.one_hot_features = []

    def processing_missing_values(self, to='auto', features=None) :
        """
        Данный функция обрабатывает пропущенные значения
        Способы обработки:
        *auto - если количество пропущенных значений больше 80%, то признак удаляется, иначе заменяем
            пропущенные значения на средние арифметическое, а если это столбец с категориями, то замена на
            часто встречаемое значение
        *mean - заменяем пропущенные значения на средние арифметическое
        *median - заменяем пропущенные значения на медиану
        *most_frequent - заменяем пропущенные значения на часто встречаемые
        *del - удаление объекта с пропущенным значением
            #TODO сделать не только для NaN
        :param to: способ обработки
        :param features: индексы признаков которые надо обработать
        :return: обработанный X
        """
        base_null = ['null', 'NULL', 'NaN', 'nan', '-', '?']
        if features is None :
            features = range(len(self.np_dataset[0]))
        if to == 'auto' :
            index_no_del_features = []
            for feature in features :
                index_missing_values = []
                index_filled_values = []
                this_column_categorical = False
                for sample in range(len(self.np_dataset)) :
                    if self.np_dataset[sample, feature] in base_null \
                            or type(self.np_dataset[sample, feature]) != str :
                        if self.np_dataset[sample, feature] in base_null or np.isnan(self.np_dataset[sample, feature]) :
                            index_missing_values.append(sample)
                        else :
                            index_filled_values.append(sample)
                    else:
                        this_column_categorical = True
                        index_filled_values.append(sample)
                if 1 - len(index_missing_values) / len(self.np_dataset) > 0.2 :
                    if len(index_filled_values) != 0 :
                        if this_column_categorical:
                            value = mode(self.np_dataset[np.array(index_filled_values), feature])[0][0]
                        else:
                            value = self.np_dataset[np.array(index_filled_values), feature].mean()
                    if len(np.array(index_missing_values)) != 0 :
                        self.np_dataset[np.array(index_missing_values), feature] = value
                    index_no_del_features.append(feature)
            self.np_dataset = self.np_dataset[:, index_no_del_features]
        elif to == 'mean' or to == 'median' or to == 'most_frequent' :
            for feature in features :
                index_missing_values = []
                index_filled_values = []
                for sample in range(len(self.np_dataset)) :
                    if self.np_dataset[sample, feature] in base_null or np.isnan(self.np_dataset[sample, feature]) :
                        index_missing_values.append(sample)
                    else :
                        index_filled_values.append(sample)
                if 1 - len(index_missing_values) / len(self.np_dataset) > 0.2 :
                    if to == 'mean' :
                        value = self.np_dataset[np.array(index_filled_values), feature].mean()
                    elif to == 'median' :
                        value = np.median(self.np_dataset[np.array(index_filled_values), feature])
                    elif to == 'most_frequent' :
                        value = mode(self.np_dataset[np.array(index_filled_values), feature])[0][0]
                    if len(np.array(index_missing_values)) != 0 :
                        self.np_dataset[np.array(index_missing_values), feature] = value
        else :
            full_dataset = np.concatenate((self.np_dataset, self.target[:, None]), axis=1)
            full_dataset_without_nan = np.array(pd.DataFrame(full_dataset).dropna())
            # full_dataset_without_nan = np.where(full_dataset_without_nan not in base_null)
            self.np_dataset = full_dataset_without_nan[:, :-1]
            self.target = full_dataset_without_nan[:, -1 :]
        return self.np_dataset

    def handling_outliners(self, method=None, factor=3, features=None) :
        """
        Данная функция обрабатывает выбросы
        Способы обработки:
        *std - удаление объекта, если его признак выходит за рамки +-(feature.all()).std()*factor
        *percentile - удаление объекта, если его признак меньше процентиля 0.05 или больше процентиля 0.95
        *None - заменяет признак объекта на процентиль 0.95, если он его больше, или заменяет на процентиль
            0.05, если он его меньше
        :param method: метод обработки
        :param factor: коэфициент смещения стандратного отклонения
        :param features: индексы признаков которые надо обработать
        :return: обработанный X
        """

        if features is None :
            features = range(len(self.np_dataset[0]))
        for feature in features :
            if method == 'std' :
                upper_lim = self.np_dataset[:, feature].mean() + self.np_dataset[:, feature].std() * factor
                lower_lim = self.np_dataset[:, feature].mean() - self.np_dataset[:, feature].std() * factor
            else :
                upper_lim = np.quantile(self.np_dataset[:, feature], .95)
                lower_lim = np.quantile(self.np_dataset[:, feature], .05)

            if method == 'percentile' or method == 'std' :
                self.np_dataset = self.np_dataset[(self.np_dataset[:, feature] < upper_lim) &
                                                  (self.np_dataset[:, feature] > lower_lim)]
            else :
                self.np_dataset[(self.np_dataset[:, feature] > upper_lim), feature] = upper_lim
                self.np_dataset[(self.np_dataset[:, feature] < lower_lim), feature] = lower_lim
        return self.np_dataset

    def binning(self, n_bins, type_binning='equal', features=None) :
        """
        Данная функция проводит категоризацию значений признака
        Способы категоризации:
        *equal - разделение на одинаковые категории по размерам
        *entropy - разделение на категории с максимальной информативностью по отношению к Y
        *quantile - разделение на категории с помощью кватилий
        :param n_bins: насколько категорий разделить
        :param type_binning: способ категоризации
        :param features: индексы признаков которые надо обработать
        :return: обработанный X
        """
        if features is None :
            features = range(len(self.np_dataset[0]))
        for feature in features :
            if type_binning == 'equal' :
                self.np_dataset[:, feature] = binning.equal_width_binning(self.np_dataset[:, feature], n_bins, False) \
                    .astype(str)
            elif type_binning == 'entropy' :
                self.np_dataset[:, feature] = binning.entropy_binning(self.np_dataset[:, feature], self.target, n_bins,
                                                                      False).astype(str)
            else :
                self.np_dataset[:, feature] = binning.quantile_binning(self.np_dataset[:, feature], n_bins, False) \
                    .astype(str)

            return self.np_dataset

    def transform(self, type_transform='log', arg=10, features=None) :
        """
        Данная функция проводит трансформацию признака с помощью определнных функций
        Способы трансформации:
        *log - логарифмирует признак
        *box-cox - трансформирует признак методом Бокса-Кокса
        :param type_transform: способ трансофрмации
        :param arg: основание логарифма для способа log или лямбда для способа box-cox
        :param features: индексы признаков которые надо обработать
        :return: обработанный X
        """
        if features is None :
            features = range(len(self.np_dataset[0]))
        for feature in features :
            if type(self.np_dataset[0, feature]) != str :
                if type_transform == 'log' :
                    self.np_dataset[:, feature] = to_log(self.np_dataset[:, feature], arg)
                elif type_transform == 'box-cox' :
                    self.np_dataset[:, feature] = to_box_cox(self.np_dataset[:, feature], arg)
            return self.np_dataset

    def scaling(self, type_scale='norm', features=None) :
        """
        Данная функция уменьшает значения признаков
        Способы уменьшения:
        *norm - нормализирует значения признаков
        *stand - стандартизирует значения признаков
        *l2-norm - делит значения признака на норму этого признака
        :param type_scale: способ уменьшения
        :param features: индексы признаков которые надо обработать
        :return: обработанный X
        """
        if features is None :
            features = range(len(self.np_dataset[0]))
        for feature in features :
            if type(self.np_dataset[0, feature]) != str :
                if type_scale == 'norm' :
                    self.np_dataset[:, feature] = scaling.normalization(self.np_dataset[:, feature])
                elif type_scale == 'stand' :
                    self.np_dataset[:, feature] = scaling.standardization(self.np_dataset[:, feature])
                elif type_scale == 'l2-norm' :
                    self.np_dataset[:, feature] = scaling.l2_normalized(self.np_dataset[:, feature])
            return self.np_dataset

    def one_hot_check(self):
        """
        Данная функция проверяет признака на то, чтобы они не были категориальными
        :return: индексы признаков с категориями
        """
        self.one_hot_features = []
        for index in range(len(self.np_dataset[0])) :
            if type(self.np_dataset[0, index]) == str :
                self.one_hot_features.append(index)
        return self.one_hot_features

    def preprocessing_manager(self, kwargs) :
        """
        Данная функция вызывает те функции данного класса, которые были заданны в словаре kwargs.
        В конце проверяет на то, какие признаки имеют категории и записывает их индексы
            в self.one_hot_features
        Формат задачи kwargs:
        {'func1':{'arg1_1':value,'arg1_2':value,...}, 'func2':{'arg2_1':value,'arg2_2':value,...},...}
        :param kwargs: словарь вызываемых функций
        :return: обработанный X
        """
        if 'processing_missing_values' in kwargs :
            if 'to' not in kwargs['processing_missing_values'] :
                kwargs['processing_missing_values']['to'] = 'auto'
            if 'features' not in kwargs['processing_missing_values'] :
                kwargs['processing_missing_values']['features'] = None
            self.processing_missing_values(kwargs['processing_missing_values']['to'],
                                           kwargs['processing_missing_values']['features'])
        if 'handling_outliners' in kwargs :
            if 'method' not in kwargs['handling_outliners'] :
                kwargs['handling_outliners']['method'] = None
            if 'factor' not in kwargs['handling_outliners'] :
                kwargs['handling_outliners']['factor'] = 3
            if 'features' not in kwargs['handling_outliners'] :
                kwargs['handling_outliners']['features'] = None
            self.handling_outliners(kwargs['handling_outliners']['method'], kwargs['handling_outliners']['factor'],
                                    kwargs['handling_outliners']['features'])
        if 'binning' in kwargs :
            if 'n_bins' not in kwargs['binning'] :
                kwargs['binning']['n_bins'] = 3
            if 'type_binning' not in kwargs['binning'] :
                kwargs['binning']['type_binning'] = 'equal'
            if 'features' not in kwargs['binning'] :
                kwargs['binning']['features'] = None
            self.binning(kwargs['binning']['n_bins'], kwargs['binning']['type_binning'], kwargs['binning']['features'])
        if 'transform' in kwargs :
            if 'type_transform' not in kwargs['transform'] :
                kwargs['transform']['type_transform'] = 'log'
            if 'arg' not in kwargs['transform'] :
                kwargs['transform']['arg'] = 10
            if 'features' not in kwargs['transform'] :
                kwargs['transform']['features'] = None
            self.transform(kwargs['transform']['type_transform'], kwargs['transform']['arg'],
                           kwargs['transform']['features'])
        if 'scaling' in kwargs :
            if 'type_scale' not in kwargs['scaling'] :
                kwargs['scaling']['type_scale'] = 'norm'
            if 'features' not in kwargs['scaling'] :
                kwargs['scaling']['features'] = None
            self.scaling(kwargs['scaling']['type_scale'], kwargs['scaling']['features'])
        self.one_hot_check()
        return self.np_dataset

    def get_dataframe(self):
        return pd.DataFrame(np.concatenate((self.np_dataset, self.target[:,None]),axis=1))

    def one_hot_encoder_categorical_features(self):
        self.one_hot_check()
        no_one_hot_features = []
        for index in range(len(self.np_dataset[0])):
            if index not in self.one_hot_features:
                no_one_hot_features.append(index)
        enc = OneHotEncoder()
        transformed_features = enc.fit_transform(self.np_dataset[:, np.array(self.one_hot_features)]).toarray()
        new_dataset = np.concatenate((self.np_dataset[:, np.array(no_one_hot_features)], transformed_features), axis=1)
        return new_dataset



if __name__ == '__main__' :
    # a = np.random.rand(100).reshape(10, 10)*100
    # b = np.random.randint(0, 2, (10,1))
    # dataset = np.concatenate((a,b),axis=1)
    # print(dataset)

    # q = pp.binning(3, features=[0])
    # print(q)
    # df = pd.read_csv('../datasets/housing.csv')
    a = np.array([[1, 2, 'o', 1], [3, 4, 'p', 0], [np.nan, 0, np.nan, 0], [4, 100, '?', 1]], dtype=np.object)
    pp = PreProcessing(a, -1)
    print(pp.np_dataset)
    rules = {'processing_missing_values' : {'to' : 'del'}, 'handling_outliners' : {'method' : None, 'features' : [1]},
             'binning' : {'n_bins' : 3, 'features' : [1]}, 'transform' : {}}
    pp.preprocessing_manager(rules)
    print(pp.np_dataset)
    print(pp.one_hot_encoder_categorical_features())
    print(pp.target)
