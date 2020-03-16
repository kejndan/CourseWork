from best_solution.settings import MEDIA_ROOT
from construction_pipeline.genetic_algorithm import GeneticClassification, GeneticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil, os
import  numpy as np

def get_names(number, exception=[]):
    names = []
    for i in range(number):
        if exception != []:
            if i in exception:
                names.append('Feature {0}'.format(i))
        else:
            names.append('Feature {0}'.format(i))
    return np.array(names)


def handle_uploaded_file(f):
    with open(MEDIA_ROOT+'/data.csv', 'wb+') as dest:
        for chunk in f.chunks():
            dest.write(chunk)

def remove_folder_contents(folder):
    for the_file in os.listdir(folder) :
        file_path = os.path.join(folder, the_file)
        try :
            if os.path.isfile(file_path) :
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e :
            print(e)


