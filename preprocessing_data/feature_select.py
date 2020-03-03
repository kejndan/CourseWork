import numpy as np
from collections import Counter
from preprocessing_data.binning import _entropy
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tpot import TPOTClassifier
from time import time


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
    # clf = Lasso()
    # a = np.array([[x,x] for x in range(30)])
    # b = np.array([-x + 20 for x in range(30)])
    # clf.fit(a,b)
    # x,y = a.shape
    # names = np.arange(y)
    # val = sorted(zip(map(lambda x: round(x, 4), clf.scores_),
    #                  names), reverse=True)
    # print(val)
    df = pd.read_csv('../datasets/sonar.csv')
    # df = df.drop(df.index[200 :])
    # x_train, x_test, y_train, y_test = train_test_split(df)
    # print(df.transpose)
    # df = np.transpose(df)
    # df = df.dropna
    print(df)
    FS = FeatureSelect(np.array(df.drop(df.columns[-1], 1)), np.array(df[df.columns[-1]]))
    print(FS.data, FS.labels)
    data = FS.main(.1, .5)

    df = pd.DataFrame(data)
    print(df)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1),df[df.columns[-1]],test_size=.2,random_state=42)
    s = time()
    tpot = TPOTClassifier(generations=5,population_size=30, verbosity=2, n_jobs=1)
    tpot.fit(x_train,y_train)
    print(time()-s)
    print(tpot.score(x_test, y_test))
    # X = np.array(df.drop(df.columns[-1], 1))
    # y = np.array(df[df.columns[-1]])
    # x1 = np.array(train_set.drop(train_set.columns[-1],1))
    # y1 = np.array(train_set[train_set.columns[-1]])
    # x2 = np.array(test_set.drop(test_set.columns[-1], 1))
    # y2 = np.array(test_set[test_set.columns[-1]])
    # print(x1.shape,y1.shape,x2.shape,y2.shape)
    # # n_samples, n_features = x1.shape
    # rng = SVR(C=1.0, epsilon=.2)
    # rng2 = SVR(C=1.0, epsilon=.2)
    # rng.fit(x1,y1)
    # # print(x1.shape,y1.shape)
    # print(rng.score(x2,y2))
    # # a = np.array([[x,x,x] for x in range(100)])
    # # b = np.array([x for x in range(100)])
    # # print(x1,y1)
    # FS = FeatureSelect(X,y)
    # print(X.shape,y.shape)
    # x3,y3,x4,y4 = FS.main(.1,0.5)
    # print(x3.shape, y3.shape, x4.shape, y4.shape)
    # print('x3',x3.shape)
    # rng2.fit(x3,y3)
    # print(rng2.score(x4,y4))
    # # print('x3',x3)
    # # rng.fit(x3,y1)
    # # FS1 = FeatureSelect(x2,y2)
    # # x4 = FS1.main(.1,.5)
    # # print('x4',x4.shape)
    # # print(rng.score(x3,y1))
    # # q = FS.main(.5,.6)
    # # print(q)
    # # FS = FeatureSelect(np.array([1,1,1,1,0,0,0,1,0,0,0,1,0,1]),np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0]))
    # # outlook = np.array([0,0,1,2,2,2,1,0,0,2,0,1,1,2])
    # # temperature = np.array([0,0,0,1,2,2,2,1,2,1,1,1,0,1])
    # # # print(FS.preprocessing_data(temperature))
    # # a = np.array([1,2,3,4,5,6])
    # # b = np.array([1,4,9,16,25,36])
    # # a1 = np.array([1,2,3,4,5])
    # # b1 = np.array([-1,-2,-3,-4,-5])
    # # print(FS.dist_corr(a1,b1))
    # # print(1-correlation(a1,b1))
    #
