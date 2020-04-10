from sklearn import ensemble, feature_selection
from sklearn.pipeline import make_pipeline
import sklearn
from preprocessing_data.preprocessing import PreProcessing
import pandas as pd
import numpy as np

q = make_pipeline(*[sklearn.preprocessing.StandardScaler(), sklearn.ensemble.GradientBoostingRegressor(alpha=0.8,learning_rate=0.1,loss='huber',max_depth=10,max_features=0.6000000000000001,min_samples_leaf=15,min_samples_split=17,n_estimators=100,subsample=0.8500000000000001)])
name = 'datasets/regression/house_prices.csv'

def fast_exp(list_models):
    pipeline = []
    for name, d in list_models :
        func = name + '('
        for k, v in d.items() :
            if k != 'name_transform' and k!='type_transform':
                func += f'{k}={v},'
        func = func[:-1]
        func += ')'
        pipeline.append(func)
    print(pipeline)


list_models = [('sklearn.preprocessing.StandardScaler', {'name_transform': 'sklearn.preprocessing.StandardScaler', 'type_transform': 'preprocessing'}),
('sklearn.ensemble.GradientBoostingRegressor', {'alpha': 0.8, 'learning_rate': 0.1, 'loss': 'huber', 'max_depth': 10, 'max_features': 0.6000000000000001, 'min_samples_leaf': 15, 'min_samples_split': 17, 'n_estimators': 100, 'name_transform': 'sklearn.ensemble.GradientBoostingRegressor', 'subsample': 0.8500000000000001, 'type_transform': 'regression'})
]

fast_exp(list_models)
information_work_sam = [[], []]
information_work_tpot = [[], []]
df = pd.read_csv(name)
# df = df.drop(df.index[10000 :])
df = df.drop(df.columns[0],1)
pp = PreProcessing(df, -1)
pp.processing_missing_values()
pp.one_hot_encoder_categorical_features()
df = pp.get_dataframe()
print(df)
x_train, y_train = df.drop(df.columns[-1], 1), df[df.columns[-1]]
q.fit(x_train, y_train)
name2 = 'datasets/regression/test.csv'
df = pd.read_csv(name2)
# df = df.drop(df.index[10000 :])
df2 = df.drop(df.columns[0],1)
pp1 = PreProcessing(df2, None)
pp1.enc = pp.enc
pp1.processing_missing_values()
pp1.one_hot_encoder_categorical_features()
df2 = pp1.get_dataframe()
print(df2)
a = q.predict(df2)
print(np.array(df2))
print(a)

a = a[:,None]
print()
a = np.concatenate((np.array(df[df.columns[0]])[:,None],a),axis=1)
a = a.astype(np.object)
a[:,0] = np.int32(a[:,0])
print(a)
pd.DataFrame(a).to_csv('mycsvfile3.csv', index=False)
