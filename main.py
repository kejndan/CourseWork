# coding=utf8
import pandas as pd
import os
import tarfile
from six.moves import urllib
from matplotlib import pyplot as plt
import numpy as np
from preprocessing_data import feature_select, preprocessing
from construction_pipeline import genetic_algorithm
from sklearn.model_selection import train_test_split
from time import time
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_log_error


if __name__ == '__main__':
    name = 'regression/diamonds.csv'

    information_work_sam = [[], []]
    information_work_tpot = [[], []]
    df = pd.read_csv('datasets/' + name)
    df = df.drop(df.index[10000 :])
    # df = df.drop(df.columns[4],1)
    pp = preprocessing.PreProcessing(df, -4)
    pp.processing_missing_values()
    pp.one_hot_encoder_categorical_features()
    df = pp.get_dataframe()
    print(df)
    # X, y = np.array(df.drop(df.columns[-1], 1)), np.array(df[df.columns[-1]])
    # FS = feature_select.FeatureSelect(X[:, :5], y, no_change_features=X[:, 5 :12])
    # x_train, y_train, x_test, y_test = FS.main(.1, 0.4)
    # print(pd.DataFrame(x_train))
    # print(pd.DataFrame(y_train))
    for i in range(5) :
        x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2)
        # GB = GeneticClustering(population_size=30, n_generations=5, name=name)
        # GB.cv = 3
        s = time()
        GB = genetic_algorithm.GeneticRegression(population_size=50, n_generations=5, name=name)
        GB.cv = 3
        # GB.cv_func = 'neg_mean_squared_log_error'
        # GB.score_func = mean_squared_log_error
        GB.fit(x_train, y_train)
        GB.score(x_test, y_test, time() - s, information_work_sam, cross_val=False)
        # print(time()-s)
        t1 = information_work_sam[1][-1]
        res1 = information_work_sam[0][-1]
        print(res1)
        print(t1)
        # print(np.array(information_work[0]).mean())
        # print(np.array(information_work[1]).mean())
        tpotr = TPOTRegressor(generations=5, population_size=30, verbosity=2, n_jobs=1)
        s = time()
        y_train = y_train.astype(np.int)
        tpotr.fit(x_train, y_train)
        t2 = time() - s
        res2 = tpotr.score(x_test, y_test)
        print(res2)
        print(t2)
        # print('-'*100)
        # information_work_tpot[0].append(res2)
        # information_work_tpot[1].append(t2)
        # with open('new_'+name[:name.rfind('.')] + '_stats.txt', 'a') as f :
        #     f.write('MyAlg ' + str(res1) + ' ' + str(t1) + '\n')
        #     f.write('TPOTAlg ' + str(res2) + ' ' + str(t2) + '\n')
    print(np.array(information_work_sam[0]).mean())
    print(np.array(information_work_sam[1]).mean())
    print(information_work_sam[0])
    print(information_work_sam[1])
    # print(np.array(information_work_tpot[0]).mean())
    # print(np.array(information_work_tpot[1]).mean())
    # with open('new_' + name[:name.rfind('.')] + '_stats.txt', 'a') as f :
    #     f.write('\n')
    #     f.write('MyAlg ' + str(np.array(information_work_sam[0]).mean()) + ' ' + str(np.array(information_work_sam[1]).mean()) + '\n')
    #     f.write('TPOTAlg ' + str(np.array(information_work_tpot[0]).mean()) + ' ' + str(np.array(information_work_tpot[1]).mean()) + '\n')
