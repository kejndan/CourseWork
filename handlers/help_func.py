from construction_pipeline.genetic_algorithm import GeneticClassification
from sklearn.model_selection import train_test_split
import pandas as pd

def genetic_alg(filename):
    df = pd.read_csv(filename)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2,
                                                        random_state=42)
    GB = GeneticClassification(population_size=30, n_generations=2, name=filename)
    GB.cv = 3
    GB.fit(x_train, y_train)
    GB.score(x_test, y_test, 10)