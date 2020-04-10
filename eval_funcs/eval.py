# coding=utf8
import sklearn
from sklearn import ensemble
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
import pandas as pd
from preprocessing_data import feature_select, preprocessing



def draw_graph(name):
    errors = []
    with open('../results/eval_params_models/'+name,'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            errors.append(list(map(lambda x: np.float(x[:-1]),line[1:-1].split())))
            line = f.readline()
            line = f.readline()
    errors = np.array(errors)
    mean_errors = []
    for column in range(len(errors[0,:])):
        mean_errors.append(np.nanmean(errors[:, column]))
    # with open('../results/eval_params_models/KNeighborsRegressor.txt', 'a') as f :
    #     f.write('\n\n')
    #     f.write(f'{mean_errors}')
    criterion = np.array(['mse', 'mae'])
    max_depth = list(range(1, 13))
    min_samples_split = list(range(2, 20))
    min_samples_leaf = list(range(1, 20))
    # p = np.array([1, 2])
    all_comb = list(product(criterion, max_depth, min_samples_split, min_samples_leaf))
    indexes = []
    for i, comb in enumerate(all_comb):
        if comb[0]=='mse' and 2<comb[1] <8 and comb[3] < 5 and comb[2] < 20 or True:
            indexes.append(i)
    indexes = np.array(indexes)
    mean_errors = np.array(mean_errors)

    plt.plot(indexes, mean_errors[indexes], 'bo', markersize=2)
    plt.xlabel('Номер комбинации')
    plt.ylabel('Точность '+'$R^2$')
    plt.title(f'Decision tree regressor')
    plt.ylim(0,.6)
    plt.show()


if __name__ == '__main__':
    n_estimators = list(range(100, 101))
    criterion = np.array(['mse', 'mae'])
    max_features = list(np.arange(0.3, 0.75, 0.05))
    min_samples_split = list(range(2, 30))
    min_samples_leaf = list(range(1, 11))
    bootstrap = np.array([True, False])
    # p = np.array([1, 2])
    all_comb = list(product(criterion, bootstrap, n_estimators, max_features, min_samples_split, min_samples_leaf))
    print(len(all_comb))
    names = ['automobile.data', 'boston_data.csv', 'communities.data', 'Daily_Demand_Forecasting_Orders.csv',
             'dataset_Facebook.csv', 'Fish.csv',
             'fundamentals.csv', 'house_prices.csv', 'parkinsons.data', 'student-mat.csv',
             'tracks.csv', 'tracks_plus.csv',
             'winequality_red.csv', 'winequality_white.csv', 'winequalityN.csv']
    for part_name in names:
        name = '/regression/' + part_name

        df = pd.read_csv('../datasets' + name)
        # df = df.drop(df.index[26000 :])
        # df = df.drop(df.columns[4],1)
        pp = preprocessing.PreProcessing(df, -1)
        pp.processing_missing_values()
        pp.one_hot_encoder_categorical_features()
        df = pp.get_dataframe()
        print(df)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df.drop(df.columns[-1], 1),
                                                                                    df[df.columns[-1]], test_size=.2,
                                                                                    random_state=35)
        numb = []
        errors = []
        for i, comb in enumerate(all_comb) :
            if comb[1] != "distance" or True :
                if i % 250 == 0 :
                    print(i)
                trans = ensemble.RandomForestRegressor(criterion=comb[0], bootstrap=comb[1],n_estimators=comb[2],
                                                       max_features=comb[3],
                                                       min_samples_split=comb[4], min_samples_leaf=comb[5])
                try :
                    trans.fit(x_train, y_train)
                    err = sklearn.metrics.r2_score(y_test, trans.predict(x_test))
                except Exception :
                    err = np.nan
                numb.append(i)
                errors.append(err)

        with open('../results/eval_params_models/RandomForestRegressor.txt', 'a') as f :
            f.write('\n')
            f.write(f'{part_name}')
            f.write(f'{errors}')
        plt.plot(numb, errors, 'bo', markersize=1)
        plt.xlabel('Номер коомбинации')
        plt.ylabel('Точность')
        plt.title(f'{name[name.rfind("/") + 1 :]}')
        plt.show()
    # draw_graph('DecisionTreeRegressor.txt')



