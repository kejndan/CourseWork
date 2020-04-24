from best_solution.settings import MEDIA_ROOT
import os
import numpy as np


def get_names(number, exception=None):
    if exception is None:
        exception = []
    names = []
    for i in range(number):
        if exception != []:
            if i in exception:
                names.append('Feature {0}'.format(i))
        else:
            names.append('Feature {0}'.format(i))
    return np.array(names)


def handle_uploaded_file(f, name):
    with open(MEDIA_ROOT+'/'+name, 'wb+') as dest:
        for chunk in f.chunks():
            dest.write(chunk)


def remove_folder_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def prepare_for_json(name, *dicts):
    output_dict = {'Dataset':{}}
    if name == 'Dataset':
        for name_column in dicts[0].keys():
            if dicts[1][name_column]:
                output_dict[name][name_column] = 'Target'
            else:
                output_dict[name][name_column] = dicts[0][name_column]
    return output_dict
