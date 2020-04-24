# coding=utf8
import sklearn
from sklearn import ensemble
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
import pandas as pd
from preprocessing_data import feature_select, preprocessing
from skopt import BayesSearchCV


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

def add_list_to_dict(list_params, name,dict_with_params):
    dict_with_params[name] = dict()
    for elem in list_params:
        dict_with_params[name][elem] = 0
    return dict_with_params


if __name__ == '__main__':
    dict_with_params = dict()
    max_features = list(np.arange(0.1, 1.01, 0.05))
    add_list_to_dict(max_features,'max_features',dict_with_params)
    min_samples_split = list(np.arange(2,30))
    add_list_to_dict(min_samples_split, 'min_samples_split',dict_with_params)
    min_samples_leaf = list(np.arange(1,30))
    add_list_to_dict(min_samples_leaf, 'min_samples_leaf',dict_with_params)
    all_comb = list(product(max_features,min_samples_split, min_samples_leaf))

    names = ('max_features','min_samples_split', 'min_samples_leaf')
    print(len(all_comb))
    names = ['automobile.data','boston_data.csv','communities.data','Daily_Demand_Forecasting_Orders.csv',
             'dataset_Facebook.csv','Fish.csv','forestfires.csv','fundamentals.csv','house_prices.csv',
             'parkinsons.data','student-mat.csv','tracks.csv','tracks_plus.csv','winequality_red.csv',
             'winequality_white.csv','winequalityN.csv']
    dist = {'max_features':max_features,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

    for part_name in names :
        name = '/classification/' + part_name
        # name = '/regression/diamonds.csv'
        # batch_comb = random.choices(all_comb, k = int(len(all_comb)*0.1))
        df = pd.read_csv('/content/drive/My Drive/datasets' + name)
        print(df.columns)
        # df = df.drop(df.index[26000 :])
        # df = df.drop(df.columns[4],1)
        pp = PreProcessing(df, -1)
        pp.processing_missing_values()
        pp.one_hot_encoder_categorical_features()
        df = pp.get_dataframe()
        df = df.astype(float)
        # df.columns = [str(s) for s in df.columns]
        print(df)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df.drop(df.columns[-1], 1),
                                                                                    df[df.columns[-1]], test_size=.2,
                                                                                    random_state=35)
        numb = []
        errors = []
        count = 0
        trans = XGBRegressor()
        src = BayesSearchCV(trans, dist, scoring='f1_score', n_jobs=-1)
        search = src.fit(x_train, y_train)
        res = dict()
        count = dict()
        for i in range(len(search.cv_results_['params'])) :
            args = (
                search.cv_results_['params']['max_features'],
                search.cv_results_['params'][i]['min_samples_split'],
                search.cv_results_['params'][i]['min_samples_leaf'],
            )
            # print(args)
            val = search.cv_results_['mean_test_score'][i]
            if all_comb.index(args) not in res :
                res[all_comb.index(args)] = 0
                count[all_comb.index(args)] = 0
            print(i)
            res[all_comb.index(args)] += val
            count[all_comb.index(args)] += 1
        for k in res.keys() :
            res[k] /= count[k]