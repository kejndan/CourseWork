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


def prepare_for_json(names, dicts):
    output_dict = dict()
    for i, name in enumerate(names):
        output_dict[name] = {}
        if name == 'Dataset':
            for name_column in dicts[i][1].keys():
                if dicts[i][1][name_column]:
                    output_dict[name][name_column] = 'Target'
                else:
                    output_dict[name][name_column] = dicts[i][0][name_column]
        else:
            output_dict[name] = dicts[i]
    return output_dict
