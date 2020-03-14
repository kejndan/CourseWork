from construction_pipeline.genetic_algorithm import GeneticClassification, GeneticRegression, GeneticClustering
from sklearn.model_selection import train_test_split
import pandas as pd


def file_to_alg(path, filename):
    df = pd.read_csv(path+filename)
    df = df.drop(df.columns[0], 1)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2,
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
    df = df.drop(df.columns[0], 1)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2)
    GB = get_solver(class_problems)(population_size=50, n_generations=3, name=filename, path=path)
    print(GB)
    GB.cv = 3
    GB.fit(x_train, y_train)
    GB.score(x_test, y_test, 10, [[],[]])
    return None
