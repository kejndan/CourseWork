from construction_pipeline.genetic_algorithm import GeneticClassification, GeneticRegression, GeneticClustering
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np
from preprocessing_data.preprocessing import PreProcessing
import pickle


def file_to_alg(path, filename):
    df = pd.read_csv(path+filename)
    # df = df.drop(df.columns[0], 1)
    with open(path+'\info_algorithm.json') as file:
        info_data = json.load(file)
    features = df.copy()
    for number, name in enumerate(info_data.keys()):
        features = features.drop(df.columns[number], 1)
        if info_data[name] == 'Target':
            target = df[df.columns[number:number+1]]


    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2,
                                                        random_state=42)
    GB = GeneticRegression(population_size=50, n_generations=3, name=filename, path=path)
    GB.cv = 3
    GB.fit(x_train, y_train)
    GB.score(x_test, y_test, 10, [[],[]])
    return None

def get_solver(class_problems):
    if class_problems == 'regression':
        return GeneticRegression
    elif class_problems == 'classification':
        return GeneticClassification
    elif class_problems == 'clustering':
        return GeneticClustering

def algorithm_manager(path, filename, class_problems):
    df = pd.read_csv(path + filename)
    with open(path + '\info_algorithm.json') as file:
        info_data = json.load(file)
    features = df.copy()
    new_names = []
    for number, name in enumerate(info_data['Dataset'].keys()):
        if not info_data['Dataset'][name]:
            features = features.drop(df.columns[number], 1)

        if info_data['Dataset'][name] == 'Target':
            features = features.drop(df.columns[number], 1)
            target = df[df.columns[number :number + 1]]
    preprocessor = PreProcessing(pd.concat([features, target], axis=1), -1)
    preprocessor.one_hot_check()
    if np.any(np.array(list(info_data['Processing_missing'].values()))):
        try:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Processing_missing'][features.columns[i]]])
        except Exception:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Processing_missing'][f'Feature {i}']])
        preprocessor.processing_missing_values(features=changed_features)
    if np.any(np.array(list(info_data['Handling_outliners'].values()))):
        try:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Handling_outliners'][features.columns[i]]])
        except Exception :
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Handling_outliners'][f'Feature {i}']])
        preprocessor.handling_outliners(features=changed_features)
    if np.any(np.array(list(info_data['Binning'].values()))):
        try:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Binning'][features.columns[i]]])
        except Exception:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Binning'][f'Feature {i}']])
        preprocessor.binning(info_data['Number_bins'],features=changed_features)
    if np.any(np.array(list(info_data['Transform'].values()))):
        try:
            changed_features = np.array([i for i in range(len(features.columns)) if info_data['Transform'][features.columns[i]]])
        except Exception:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Transform'][f'Feature {i}']])
        preprocessor.transform(info_data['Type_transform'],features=changed_features)
    if np.any(np.array(list(info_data['Scaling'].values()))):
        try:
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Scaling'][features.columns[i]]])
        except Exception :
            changed_features = np.array(
                [i for i in range(len(features.columns)) if info_data['Scaling'][f'Feature {i}']])
        preprocessor.scaling(info_data['Type_scaling'],features=changed_features)
    preprocessor.one_hot_encoder_categorical_features()
    pickle.dump(preprocessor, open(path + '\preprocessor.pkl'.format(number), 'wb'))
    changed_df = preprocessor.get_dataframe()
    features = changed_df.drop(changed_df.columns[-1], 1)
    target = changed_df[changed_df.columns[-1 :]]



    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=42)
    GB = get_solver(class_problems)(population_size=30, n_generations=2, name=filename, path=path)
    GB.cv = 3
    GB.fit(x_train, np.array(y_train).reshape(-1,))
    GB.score(x_test, y_test, 10, [[],[]])
    return None
