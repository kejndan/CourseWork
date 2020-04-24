from construction_pipeline.genetic_algorithm import GeneticClassification, GeneticRegression, GeneticClustering
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np


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
        info_data = json.load(file)['Dataset']
    features = df.copy()
    for number, name in enumerate(info_data.keys()):
        if not info_data[name]:
            features = features.drop(df.columns[number], 1)
        if info_data[name] == 'Target':
            features = features.drop(df.columns[number], 1)
            target = df[df.columns[number :number + 1]]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=42)
    GB = get_solver(class_problems)(population_size=50, n_generations=3, name=filename, path=path)
    GB.cv = 3
    GB.fit(x_train, np.array(y_train).reshape(-1,))
    GB.score(x_test, y_test, 10, [[],[]])
    return None
