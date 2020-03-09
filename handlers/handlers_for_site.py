from best_solution.settings import MEDIA_ROOT
from construction_pipeline.genetic_algorithm import GeneticClassification, GeneticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


def file_to_alg(filename):
    df = pd.read_csv(filename)
    df = df.drop(df.columns[0], 1)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2,
                                                        random_state=42)
    GB = GeneticRegression(population_size=30, n_generations=5, name=filename)
    GB.cv = 3
    GB.fit(x_train, y_train)
    return GB.score(x_test, y_test, 10)


def handle_uploaded_file(f):
    with open(MEDIA_ROOT+'/data.csv', 'wb+') as dest:
        for chunk in f.chunks():
            dest.write(chunk)

