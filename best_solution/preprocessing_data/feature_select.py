# coding=utf8
from math_func import dist_corr, entropy
import numpy as np
from collections import Counter
# from tpot import TPOTClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV, LassoLarsIC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from stability_selection import RandomizedLasso
from binning import _entropy

from scipy.spatial.distance import correlation
class FeatureSelect:
    def __init__(self, change_features, labels, no_change_features=None):
        self.data = change_features
        self.labels = labels
        self.no_change_features = no_change_features
        # self.displaying_table1 = [[], []]
        # self.displaying_table2 = [[], [], []]
        # self.displaying_table3 = [[], []]

    def information_gain(self, column):
        """
        Данная функция подсчитывает информативность признака
        :param column: передаваемый признак
        :return: information_gain признака(чем больше ig тем сильнее корреляция)
        """
        count_labels = np.array(sorted(Counter(self.labels).items())).astype(np.float32)
        h_y = entropy(count_labels[:, 1], sum(count_labels[:, 1]))
        count_data = np.array(sorted(Counter(column).items())).astype(np.float32)
        data_frame = pd.DataFrame(data=[column, self.labels], dtype=np.float32).transpose()
        for i in range(len(count_data)):
            indexes = data_frame[data_frame[0] != count_data[i, 0]].index
            temp_df = data_frame.drop(indexes)
            sub_count_labels = np.array(sorted(Counter(temp_df[1]).items())).astype(np.float32)
            sub_h_y = _entropy(sub_count_labels[:, 1], sum(sub_count_labels[:, 1]))
            h_y -= count_data[i, 1] * sub_h_y / sum(count_data[:, 1])
        return h_y

    def pre_processing(self, alpha, report=True):
        """
        Данная функция удаляет лишнии признаки на основании функции information_gain
        :param alpha: порог информативности
        :param report: информация после выполненной обработки
        :return: сокращенное пространство признаков
        """
        new_data = np.empty((self.data.shape[0], 0))
        for num_feature in range(self.data.shape[1]):
            if self.information_gain(self.data[:, num_feature]) > alpha:
                # self.displaying_table1[0].append(num_feature)
                # self.displaying_table1[1].append(len(new_data[0]))
                new_data = np.hstack([new_data, self.data[:, num_feature][:, np.newaxis]])
        if report:
            print('Shape of data after PreProcessing ', new_data.shape)
        return new_data

    def feature_generation(self, data, beta, report=True):
        """
        Данная функция создает новое пространство признаков с помощью Ridge и Kernel Ridge Regression
        :param data: старое пространство признаков
        :param beta: порог distance correlation при котором мы выбираем либо RR или KRR
        :param report: информация после выполненной обработки
        :return: новое пространство признаков
        """
        new_data = np.empty((data.shape[0], 0))
        # count = 0
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                corr = 1 - correlation(data[:, i], data[:, j])
                if i != j and corr > 0:
                    if corr < beta:
                        clf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf', kernel_params=None)
                        # type_trans = 'KRR'
                    elif beta <= corr <= 1:
                        clf = Ridge(alpha=1.0)
                        # type_trans = 'KR'
                    # self.displaying_table2[0].append((i, j))
                    # self.displaying_table2[0].append((i, j))
                    # self.displaying_table2[1].append(type_trans)
                    # self.displaying_table2[1].append('-'+type_trans)
                    # self.displaying_table2[2].append(len(new_data[0]))
                    # self.displaying_table2[2].append(len(new_data[0])+1)
                    f_i = data[:, i][:, np.newaxis]
                    f_j = data[:, j]
                    f = clf.fit(f_i, f_j).predict(f_i)
                    new_data = np.hstack([new_data, f[:, np.newaxis]])
                    new_data = np.hstack([new_data, (f_j - f)[:, np.newaxis]])
                # if count % 10 == 0 :
                #     print(count, end=' ')
                # count += 1
        if report:
            print('Shape of data after feature generation ', new_data.shape)
        return new_data

    def train_test_split(self, data, test_size=0.2, return_mask=False):
        mask = np.array(np.arange(len(self.labels)))
        np.random.shuffle(mask)
        if return_mask:
            return data[mask][int(test_size * len(self.labels)):], data[mask][:int(test_size * len(self.labels))], mask
        else:
            return data[mask][int(test_size * len(self.labels)):], data[mask][:int(test_size * len(self.labels))]

    def feature_selection(self, data, alpha, report=True):
        """
        Данная функция очищает признакое пространство от слабых признаков через RandomizedLasso и
         information_gain
        :param data: признаковое пространство
        :param alpha: порог информативности
        :param report: информация после выполненной обработки
        :return: окончательное признаковое простаранство
        """
        data = np.hstack([data, self.labels[:, np.newaxis]])
        train, test, mask = self.train_test_split(data, return_mask=True)
        # x_train = np.array(train.drop(train.columns[-1], 1))
        # y_train = np.array(train[train.columns[-1]])
        # x_test = np.array(test.drop(test.columns[-1], 1))
        # y_test = np.array(test[test.columns[-1]])
        x_train = train[:, :-1]
        y_train = train[:, -1:]
        x_test = test[:, :-1]
        y_test = test[:, -1:]
        new_train, new_test = np.empty((train.shape[0], 0)), np.empty((test.shape[0], 0))
        clf = LassoCV()
        clf.fit(x_train, y_train)
        for i in range(x_train.shape[1]):
            if clf.coef_[i] >= 0.0 and self.information_gain(x_train[:, i]) > alpha:
                # self.displaying_table3[0].append(i)
                # self.displaying_table3[1].append(len(new_train[0]))
                new_train = np.hstack([new_train, x_train[:, i][:, np.newaxis]])
                new_test = np.hstack([new_test, x_test[:, i][:, np.newaxis]])
        if self.no_change_features is not None:
            new_train = np.concatenate((new_train, self.no_change_features[mask][int(0.2 * len(self.labels)):]), axis=1)
            new_test = np.concatenate((new_test, self.no_change_features[mask][:int(0.2 * len(self.labels))]), axis=1)
        if report:
            print('Shape of data after feature selection ', new_train.shape, new_test.shape)
        return new_train, y_train, new_test, y_test

    def stat_feature(self):
        stats = [0 for _ in range(len(self.data[0]))]
        for index in range(len(self.displaying_table3[1])):
                i = self.displaying_table3[0][index]
                f_pq = self.displaying_table2[2][self.displaying_table2[2].index(i)]
                p1, q1 = self.displaying_table2[0][self.displaying_table2[2].index(f_pq)]
                p = self.displaying_table1[0][self.displaying_table1[1].index(p1)]
                q = self.displaying_table1[0][self.displaying_table1[1].index(q1)]
                stats[p] += 1
                stats[q] += 1
        return stats

    def main(self, alpha, beta) :
        data = self.pre_processing(alpha)
        data = self.feature_generation(data, beta)
        data = self.feature_selection(data, alpha)
        return data


if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\adels\Documents\datasets\\classification\waveform.csv')
    # df = df.dropna(k)
    # df = df.dropna()
    # X1 = np.array(df.drop(df.columns[-1], 1))
    # df = df.drop(df.columns[0],1)
    # df = df.drop(df.index[14000:],0)
    # SS = StandardScaler()
    # MMS = MinMaxScaler()
    # df = pd.DataFrame(SS.fit_transform(df))
    # print(df.transpose)
    # df = np.transpose(df)

    print(df)
    train_set, test_set = train_test_split(df, test_size=.2, random_state=42)
    X = np.array(df.drop(df.columns[-1], 1))
    X1 = X
    # X = SS.fit_transform(X)
    y = np.array(df[df.columns[-1]])
    x1 = np.array(train_set.drop(train_set.columns[-1], 1))
    # x1 = MMS.fit_transform(SS.fit_transform(x1))
    y1 = np.array(train_set[train_set.columns[-1]])
    x2 = np.array(test_set.drop(test_set.columns[-1], 1))
    # x2 = MMS.fit_transform(SS.fit_transform(x2))
    y2 = np.array(test_set[test_set.columns[-1]])
    print(x1.shape, y1.shape, x2.shape, y2.shape)
    # n_samples, n_features = x1.shape
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    ab = AdaBoostClassifier()
    nn = MLPClassifier()
    dt = ExtraTreesClassifier()
    # knn = KNeighborsRegressor()
    # lr = LinearRegression()
    # rf = RandomForestRegressor()
    # ab = AdaBoostRegressor()
    # dt = ExtraTreesRegressor()
    # nn = MLPRegressor()
    knn.fit(x1, y1)
    lr.fit(x1, y1)
    rf.fit(x1, y1)
    ab.fit(x1, y1)
    nn.fit(x1, y1)
    dt.fit(x1, y1)
    a = knn.score(x2, y2)
    b = lr.score(x2, y2)
    c = rf.score(x2, y2)
    d = ab.score(x2, y2)
    e = nn.score(x2, y2)
    f = dt.score(x2, y2)
    with open('result.txt', 'a', encoding='utf-8') as file :
        file.write('Breast cancer\n')
        file.write('KNN {0}\n'.format(a))
        file.write('LR {0}\n'.format(b))
        file.write('RF {0}\n'.format(c))
        file.write('AB {0}\n'.format(d))
        file.write('NN {0}\n'.format(e))
        file.write('DT {0}\n'.format(f))
    from copy import deepcopy
    result = {'KNN':[],'LR':[],'RF':[],'AB':[],'NN':[],'DT':[]}
    all_stats = np.zeros(len(x1[0]))
    for i in range(1):
        # x,_ = train_test_split(df, test_size=.8)
        # xs = np.array(x.drop(x.columns[-1], 1))
        # ys = np.array(x[x.columns[-1]])
        X = np.array(X1)
        enc = OneHotEncoder()
        FS = FeatureSelect(X, y)
        # FS = FeatureSelect(xs, ys)
        x3, y3, x4, y4 = FS.main(0.1, 0.4)
        # stats = FS.stat_feature()
        # all_stats = all_stats+np.array(stats)
        old_x3 = deepcopy(x3)
        old_x4 = deepcopy(x4)
        print(x3)
        # pca = PCA(svd_solver='randomized',iterated_power=10).fit(x3)
        # x3 = pca.transform(x3)
        # x4 = pca.transform(x4)
        print()
        knn.fit(x3, y3)
        lr.fit(x3, y3)
        rf.fit(x3, y3)
        ab.fit(x3, y3)
        nn.fit(x3, y3)
        dt.fit(x3, y3)
        result['KNN'].append(knn.score(x4, y4))
        result['LR'].append(lr.score(x4, y4))
        result['RF'].append(rf.score(x4, y4))
        result['AB'].append(ab.score(x4, y4))
        result['NN'].append(nn.score(x4, y4))
        result['DT'].append(dt.score(x4, y4))
        print('KNN ', knn.score(x4, y4), 'D ', knn.score(x4, y4) - a)
        print('LR ', lr.score(x4, y4), 'D ', lr.score(x4, y4) - b)
        print('RF ', rf.score(x4, y4), 'D ', rf.score(x4, y4) - c)
        print('AB ', ab.score(x4, y4), 'D ', ab.score(x4, y4) - d)
        print('NN ', nn.score(x4, y4), 'D ', nn.score(x4, y4) - e)
        print('DT ', dt.score(x4, y4), 'D ', dt.score(x4, y4) - f)
    for k,v in result.items():
        print(k,np.array(v).mean())
    print(all_stats)
