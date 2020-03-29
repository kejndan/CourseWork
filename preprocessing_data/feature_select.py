import numpy as np
from collections import Counter
# from tpot import TPOTClassifier
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from stability_selection import RandomizedLasso
from preprocessing_data.binning import _entropy

class FeatureSelect :
    def __init__(self, data, labels) :
        self.data = data
        self.labels = labels

    def information_gain(self, column) :
        count_labels = np.array(sorted(Counter(self.labels).items())).astype(np.float32)
        h_y = _entropy(count_labels[:, 1], sum(count_labels[:, 1]))
        count_data = np.array(sorted(Counter(column).items())).astype(np.float32)
        data_frame = pd.DataFrame(data=[column, self.labels], dtype=np.float32).transpose()
        for i in range(len(count_data)) :
            indexes = data_frame[data_frame[0] != count_data[i, 0]].index
            temp_df = data_frame.drop(indexes)
            sub_count_labels = np.array(sorted(Counter(temp_df[1]).items())).astype(np.float32)
            sub_h_y = _entropy(sub_count_labels[:, 1], sum(sub_count_labels[:, 1]))
            h_y -= count_data[i, 1] * sub_h_y / sum(count_data[:, 1])
        return h_y

    def dist_corr(self, u, v) :
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
        if np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)) == 0 :
            return 0
        dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor

    def pre_processing(self, alpha, report=True) :
        new_data = np.empty((self.data.shape[0], 0))
        for i in range(self.data.shape[1]) :
            if self.information_gain(self.data[:, i]) > alpha :
                new_data = np.hstack([new_data, self.data[:, i][:, np.newaxis]])
        if report :
            print('Shape of data after PreProcessing ', new_data.shape)
        return new_data

    def feature_generation(self, data, beta, report=True) :
        new_data = np.empty((data.shape[0], 0))
        count = 0
        for i in range(data.shape[1]) :
            for j in range(data.shape[1]) :
                corr = self.dist_corr(data[:, i], data[:, j])
                if i != j and corr > 0 :
                    if corr < beta :
                        clf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                                          kernel_params=None)
                    elif beta <= corr <= 1 :
                        clf = Ridge(alpha=1.0)
                    f_i = data[:, i][:, np.newaxis]
                    f_j = data[:, j]
                    f = clf.fit(f_i, f_j).predict(f_i)
                    new_data = np.hstack([new_data, f[:, np.newaxis]])
                    new_data = np.hstack([new_data, (f_j - f)[:, np.newaxis]])
                # if count % 10 == 0 :
                #     print(count, end=' ')
                count += 1
        if report :
            print('Shape of data after feature generation ', new_data.shape)
        return new_data

    def feature_selection(self, data, alpha, report=True) :
        data = np.hstack([data, self.labels[:, np.newaxis]])
        train, test = train_test_split(pd.DataFrame(data), test_size=.2)
        x_train = np.array(train.drop(train.columns[-1], 1))
        y_train = np.array(train[train.columns[-1]])
        x_test = np.array(test.drop(test.columns[-1], 1))
        y_test = np.array(test[test.columns[-1]])
        new_train, new_test = np.empty((train.shape[0], 0)), np.empty((test.shape[0], 0))
        clf = LassoCV(cv=5, tol=0.1)
        clf = RandomizedLasso()
        clf.fit(x_train, y_train)
        for i in range(x_train.shape[1]) :
            if clf.coef_[i] > .1 and self.information_gain(x_train[:, i]) > alpha :
                new_train = np.hstack([new_train, x_train[:, i][:, np.newaxis]])
                new_test = np.hstack([new_test, x_test[:, i][:, np.newaxis]])
        if report :
            print()
            print('Shape of data after feature selection ', x_train.shape, x_test.shape)
        return x_train, y_train, x_test, y_test

    def main(self, alpha, beta) :
        data = self.pre_processing(alpha)
        data = self.feature_generation(data, beta)
        data = self.feature_selection(data, alpha)
        return data

    def old_main(self, n1, n2):
        self.new_data1 = []
        self.new_data2 = []
        self.new_data3 = []
        for i in range(self.data.shape[1]):
            if self.information_gain(self.data[:,i]) > n1:
                self.new_data1.append(self.data[:,i])
        self.new_data1 = np.transpose(np.array(self.new_data1))
        print('1',self.new_data1.shape)
        for i in range(self.new_data1.shape[1]):
            for j in range(self.new_data1.shape[1]):
                corr = self.dist_corr(self.new_data1[:,i],self.new_data1[:,j])
                if i != j and corr != 0:
                    if corr > 0 and corr < n2:
                        clf = KernelRidge(alpha=1.0)
                        f_i = self.new_data1[:,i].reshape(-1,1)
                        f_j = self.new_data1[:,j]
                        f = clf.fit(f_i,f_j).predict(f_i)
                        self.new_data2.append(f)
                        self.new_data2.append(f_j-f)
                    elif corr >= n2 and corr <=1:
                        clf = Ridge(alpha=1.0)
                        f_i = self.new_data1[:,i].reshape(-1, 1)
                        f_j = self.new_data1[:,j]
                        f = clf.fit(f_i, f_j).predict(f_i)
                        self.new_data2.append(f)
                        self.new_data2.append(f_j - f)
        self.new_data2 = np.transpose(np.array(self.new_data2))
        # print(self.new_data2)
        train, test = train_test_split(pd.DataFrame(self.new_data2), test_size=.2)
        x_train = np.array(train.drop(train.columns[-1],1))
        y_train = np.array(train[train.columns[-1]])
        x_test = np.array(test.drop(test.columns[-1], 1))
        y_test = np.array(test[test.columns[-1]])
        clf = LassoCV(cv=5,tol=0.1)
        clf.fit(x_train,y_train)
        # print(self.new_data2.shape,self.labels.shape)
        new_train = []
        new_test = []
        for i in range(x_train.shape[1]):
            # if clf.coef_[i] > .1:
            if clf.coef_[i] > .1 and self.information_gain(x_train[:,i]) > n1:
                # self.new_data3.append(self.new_data2[:,i])
                new_train.append(x_train[:, i])
                new_test.append(x_test[:,i])
        # self.new_data3 = np.transpose(np.array(self.new_data3))
        # self.new_data3 = np.array(self.new_data3)



        return np.transpose(np.array(new_train)), y_train,np.transpose(np.array(new_test)), y_test










if __name__ == '__main__':
    df = pd.read_csv('../datasets/sonar.csv')
    # df = df.dropna()
    df = df.drop(df.columns[0],1)
    # df = df.drop(df.index[14000:],0)
    # SS = StandardScaler()
    # MMS = MinMaxScaler()
    # df = pd.DataFrame(SS.fit_transform(df))
    # print(df.transpose)
    # df = np.transpose(df)
    print(df)
    train_set, test_set = train_test_split(df, test_size=.2, random_state=42)
    X = np.array(df.drop(df.columns[-1], 1))
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
    # ln = LinearRegression()
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
    for i in range(5):
        FS = FeatureSelect(X, y)
        x3, y3, x4, y4 = FS.main(.1, 0.4)
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
