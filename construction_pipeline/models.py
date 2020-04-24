import numpy as np

preprocessing_models = {
    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': np.array([2]),
        'include_bias': np.array([False]),
        'interaction_only': np.array([False])
    },
    'sklearn.preprocessing.StandardScaler': {

    },
    'sklearn.preprocessing.RobustScaler': {

    },
    'sklearn.preprocessing.MaxAbsScaler': {

    },
    'sklearn.preprocessing.MinMaxScaler': {

    },
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    }
}

selection_models = {
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]) # not change
    },
    'sklearn.feature_selection.SelectKBest': {
        'k': np.arange(1, 101)
    },
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': np.arange(1, 100)
    },
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001)
    },
    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': np.array([100]),
                'criterion': np.array(['gini', 'entropy']),
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },
    'sklearn.feature_selection.SelectFromModel' : {
        'threshold' : np.arange(0, 1.01, 0.05),
        'estimator' : {
            'sklearn.ensemble.ExtraTreesRegressor' : {
                'n_estimators' : np.array([100]),
                'max_features' : np.arange(0.05, 1.01, 0.05)
            }
        }
    }
}
classification_models = {
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': np.array(["gini", "entropy"]),
        'max_depth': np.arange(1, 11),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21)
    },
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': np.array([100]),
        'criterion': np.array(["gini", "entropy"]),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21),
        'bootstrap': np.array([True, False])
    },
    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': np.array([100]),
        'learning_rate': np.array([1e-3, 1e-2, 1e-1, 0.5, 1.]),
        'max_depth': np.arange(1, 11),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },
    'sklearn.linear_model.LogisticRegression': {
        'penalty': np.array(["l1","l2"]), # not change
        'C': np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]),
        'dual': np.array([True, False]), #not change
        'max_iter': np.array([500])
    },
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': np.arange(1, 101),
        'weights': np.array(["uniform", "distance"]),
        'p': np.array([1, 2])
    },
    'sklearn.ensemble.ExtraTreesClassifier' : {
        'n_estimators' : np.array([100]),
        'criterion' : np.array(["gini", "entropy"]),
        'max_features' : np.arange(0.05, 1.01, 0.05),
        'min_samples_split' : np.arange(2, 21),
        'min_samples_leaf' : np.arange(1, 21),
        'bootstrap' : np.array([True, False])
    },
    'sklearn.naive_bayes.GaussianNB' : {
    },
    'sklearn.naive_bayes.BernoulliNB' : {
        'alpha' : [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior' : [True, False]
    },
}

regression_models = {
    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': np.array([100]),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21),
        'bootstrap': np.array([True, False])
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': np.array([100]),
        'loss': np.array(["ls", "lad", "huber", "quantile"]),
        'learning_rate': np.array([1e-3, 1e-2, 1e-1, 0.5, 1.]),
        'max_depth': np.arange(1, 11),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': np.array([0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': np.array([100]),
        'learning_rate': np.array([1e-3, 1e-2, 1e-1, 0.5, 1.]),
        'loss': np.array(["linear", "square", "exponential"])
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': np.arange(1, 11),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': np.arange(1, 101),
        'weights': np.array(["uniform", "distance"]),
        'p': np.array([1, 2])
    },

    'sklearn.linear_model.LassoLarsCV': {
        'normalize': np.array([True, False])
    },

    'sklearn.svm.LinearSVR': {
        'loss': np.array(["epsilon_insensitive", "squared_epsilon_insensitive"]),
        'dual': np.array([True, False]),
        'tol': np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        'C': np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]),
        'epsilon': np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.])
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': np.array([100]),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21),
        'bootstrap': np.array([True, False])
    },


    'sklearn.linear_model.SGDRegressor': {
        'loss': np.array(['squared_loss', 'huber', 'epsilon_insensitive']),
        'penalty': np.array(['elasticnet']),
        'alpha': np.array([0.0, 0.01, 0.001]),
        'learning_rate':np.array(['invscaling', 'constant']),
        'fit_intercept': np.array([True, False]),
        'l1_ratio': np.array([0.25, 0.0, 1.0, 0.75, 0.5]),
        'eta0': np.array([0.1, 1.0, 0.01]),
        'power_t': np.array([0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0])
    },
}

clustering_models = {
    'sklearn.cluster.KMeans': {
        'n_clusters': np.arange(2, 11),
        # 'init': np.array(['k_means++','random'])
    },
    'sklearn.cluster.MeanShift':{

    },
    'sklearn.cluster.SpectralClustering': {
        'n_clusters': np.arange(2, 11),
    },
    'sklearn.cluster.AgglomerativeClustering': {
        'n_clusters': np.arange(2, 11),
        'linkage': np.array(['ward', 'complete', 'average'])
    },
}
