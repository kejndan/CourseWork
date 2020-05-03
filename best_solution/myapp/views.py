from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
# Imaginary function to handle an uploaded file.
from handlers.handlers_for_site import handle_uploaded_file, remove_folder_contents,get_names,prepare_for_json
from best_solution.settings import MEDIA_ROOT, THREAD
from multiprocessing import Process
from handlers.core import algorithm_manager
import os
import numpy as np
import json
import shutil
import pickle
from preprocessing_data import preprocessing


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            remove_folder_contents(MEDIA_ROOT)
            handle_uploaded_file(request.FILES['file'],'data.csv')
            return HttpResponseRedirect('/processing/')
    else:
        form = UploadFileForm()
        return render(request, 'myapp/upload.html', {'form': form})


def processing(request):
    if request.method == 'GET':

        names_for_json = []
        data_for_json = []
        df = pd.read_csv(MEDIA_ROOT+'\\data.csv')
        features = df.drop(df.columns[-1], 1)
        targets = df[df.columns[-1:]]
        try:
            names_all_features = np.array(df.columns, dtype=float)
            names_features = np.array(features.columns, dtype=float)
            names_all_features = get_names(len(names_all_features))
            names_features = get_names(len(names_features))
            name_targets = ['Target']
        except ValueError:
            names_all_features = df.columns
            names_features = features.columns
            name_targets = targets.columns

        status_checkboxes_preprocessing = dict(zip(names_features, [True for _ in range(len(names_features))]))
        status_checkboxes_handling_outliners = dict(zip(names_features, [True for _ in range(len(names_features))]))
        status_checkboxes_binning = dict(zip(names_features, [True for _ in range(len(names_features))]))
        status_checkboxes_transform = dict(zip(names_features, [True for _ in range(len(names_features))]))
        status_checkboxes_scaling = dict(zip(names_features, [True for _ in range(len(names_features))]))
        status_checkboxes = [True for _ in range(len(names_features))]
        status_checkboxes = dict(zip(names_features, status_checkboxes))
        status_radio = [False for _ in range(len(names_all_features)-1)] + [True]
        status_radio = dict(zip(names_all_features, status_radio))

        names_for_json.append('Class_problem')
        data_for_json.append('regression')
        names_for_json.append('Dataset')
        data_for_json.append((status_checkboxes, status_radio))
        names_for_json.append('Processing_missing')
        data_for_json.append(status_checkboxes_preprocessing)
        names_for_json.append('Handling_outliners')
        data_for_json.append(status_checkboxes_handling_outliners)
        names_for_json.append('Binning')
        data_for_json.append(status_checkboxes_binning)
        names_for_json.append('Transform')
        data_for_json.append(status_checkboxes_transform)
        names_for_json.append('Scaling')
        data_for_json.append(status_checkboxes_scaling)

        with open(MEDIA_ROOT + '/info_algorithm.json', 'w') as file:
            json.dump(prepare_for_json(names_for_json, data_for_json), file)
        return render(request, 'myapp/processing.html',
                      {'columns_feature': names_features, 'rows_feature': features.to_dict('records'),
                       'column_targets': name_targets, 'rows_targets': targets.to_dict('records'),
                       'status_checkboxes': status_checkboxes, 'status_radio': status_radio,
                       'on_processing_missing' : True,
                       'status_processing_missing' : status_checkboxes_preprocessing,
                       'on_handling_outliners': True,
                       'status_handling_outliners' : status_checkboxes_handling_outliners,
                       'on_binning': False,
                       'status_binning': status_checkboxes_binning,
                       'on_transform': False,
                       'status_transform': status_checkboxes_transform,
                       'on_scaling': False,
                       'status_scaling': status_checkboxes_scaling
                       })
    elif request.method == 'POST':
        names_for_json = []
        data_for_json = []
        names_for_json.append('Class_problem')
        data_for_json.append(request.POST.get("type_func"))
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')
        index_target = int(request.POST.get('checksradio[]')) - 1
        all_features = df.drop(df.columns[index_target], 1)
        targets = df[df.columns[index_target:index_target+1]]

        got_status_checkboxes_features = np.array(request.POST.getlist('checks[]'), dtype=int) - 1
        got_status_checkboxes_preprocessing = np.array(request.POST.getlist('checks_preprocessing[]'), dtype=int) - 1
        got_status_checkboxes_handling_outliners = np.array(request.POST.getlist('checks_handling_outliners[]'), dtype=int) - 1
        got_status_checkboxes_binning = np.array(request.POST.getlist('checks_binning[]'), dtype=int) - 1
        got_status_checkboxes_transform = np.array(request.POST.getlist('checks_transform[]'), dtype=int) - 1
        got_status_checkboxes_scaling = np.array(request.POST.getlist('checks_scaling[]'), dtype=int) - 1


        if np.where(got_status_checkboxes_features == index_target):
            np.delete(got_status_checkboxes_features, np.where(got_status_checkboxes_features == index_target))
        select_features = all_features.copy()
        preprocessor = preprocessing.PreProcessing(pd.concat([all_features, targets], axis=1), -1)
        preprocessor.one_hot_check()
        if 'on_processing_missing' in request.POST:
            if not np.array_equal(got_status_checkboxes_features, got_status_checkboxes_preprocessing):
                checks_feature_preprocessing = []
                for i in range(len(got_status_checkboxes_preprocessing)):
                    if got_status_checkboxes_preprocessing[i] in got_status_checkboxes_features:
                        checks_feature_preprocessing.append(got_status_checkboxes_preprocessing[i])
                got_status_checkboxes_preprocessing = np.array(checks_feature_preprocessing)
            preprocessor.processing_missing_values(features=got_status_checkboxes_preprocessing)

            # changed_df = preprocessor.get_dataframe()
            # changed_df.columns = df.columns
            # select_features = changed_df.drop(changed_df.columns[-1], 1)
            # targets = changed_df[changed_df.columns[-1:]]
            on_processing_missing = True
        else:
            on_processing_missing = False

        if 'on_handling_outliners' in request.POST:
            if not np.array_equal(got_status_checkboxes_features, got_status_checkboxes_handling_outliners):
                checks_feature_handling_outliners = []
                for i in range(len(got_status_checkboxes_handling_outliners)):
                    if got_status_checkboxes_handling_outliners[i] in got_status_checkboxes_features:
                        checks_feature_handling_outliners.append(got_status_checkboxes_handling_outliners[i])
                got_status_checkboxes_handling_outliners = np.array(checks_feature_handling_outliners)
            preprocessor.handling_outliners(features=got_status_checkboxes_handling_outliners)
            on_handling_outliners = True
        else:
            on_handling_outliners = False

        if 'on_binning' in request.POST :
            if not np.array_equal(got_status_checkboxes_features, got_status_checkboxes_binning) :
                checks_feature_binning = []
                for i in range(len(got_status_checkboxes_binning)) :
                    if got_status_checkboxes_binning[i] in got_status_checkboxes_features :
                        checks_feature_binning.append(got_status_checkboxes_binning[i])
                got_status_checkboxes_binning = np.array(checks_feature_binning)
            number_bins = int(request.POST['bins'])
            preprocessor.binning(number_bins, features=got_status_checkboxes_binning)
            on_binning = True
        else:
            on_binning = False

        if 'on_transform' in request.POST :
            if not np.array_equal(got_status_checkboxes_features, got_status_checkboxes_transform) :
                checks_feature_transform = []
                for i in range(len(got_status_checkboxes_transform)) :
                    if got_status_checkboxes_transform[i] in got_status_checkboxes_features :
                        checks_feature_transform.append(got_status_checkboxes_transform[i])
                got_status_checkboxes_transform = np.array(checks_feature_transform)
            type_transform = request.POST['type_transform']
            if type_transform == 'To Logarithm':
                type_transform = 'log'
            elif type_transform == 'To box-cox':
                type_transform = 'box-cox'
            preprocessor.transform(type_transform, features=got_status_checkboxes_transform)
            on_transform = True
        else:
            on_transform = False

        if 'on_scaling' in request.POST :
            if not np.array_equal(got_status_checkboxes_features, got_status_checkboxes_scaling) :
                checks_feature_scaling = []
                for i in range(len(got_status_checkboxes_scaling)) :
                    if got_status_checkboxes_scaling[i] in got_status_checkboxes_features :
                        checks_feature_scaling.append(got_status_checkboxes_scaling[i])
                got_status_checkboxes_scaling = np.array(checks_feature_scaling)
            type_scaling = request.POST['type_scaling']
            if type_scaling == 'Normalization' :
                type_scaling = 'norm'
            elif type_scaling == 'Standardization' :
                type_scaling = 'stand'
            elif type_scaling == 'l2-normalization' :
                type_scaling = 'l2-norm'
            preprocessor.scaling(type_scaling, features=got_status_checkboxes_scaling)
            on_scaling = True
        else:
            on_scaling = False

        changed_df = preprocessor.get_dataframe()
        changed_df.columns = df.columns
        select_features = changed_df.drop(changed_df.columns[-1], 1)
        targets = changed_df[changed_df.columns[-1:]]
        for index in range(len(all_features.columns)):
            if index not in got_status_checkboxes_features:
                select_features = select_features.drop(all_features.columns[index], 1)

        try:
            names_all_features = np.array(df.columns, dtype=float)
            names_all_features = get_names(len(names_all_features))
            names_features = np.array(all_features.columns, dtype=float)
            names_features = get_names(len(names_features))
            names_select_features = np.array(select_features.columns, dtype=float)
            names_select_features = get_names(len(names_select_features), got_status_checkboxes_features)
            name_targets = ['Target']
        except ValueError as e:
            print(e)
            names_all_features = df.columns
            names_features = all_features.columns
            names_select_features = select_features.columns
            name_targets = targets.columns
        status_checkboxes_preprocessing = []
        status_checkboxes_handling_outliners = []
        status_checkboxes_binning = []
        status_checkboxes_transform = []
        status_checkboxes_scaling = []
        for i in range(len(names_features)):
            if i in got_status_checkboxes_preprocessing and on_processing_missing:
                status_checkboxes_preprocessing.append(True)
            else:
                status_checkboxes_preprocessing.append(False)
            if i in got_status_checkboxes_handling_outliners and on_handling_outliners:
                status_checkboxes_handling_outliners.append(True)
            else:
                status_checkboxes_handling_outliners.append(False)
            if i in got_status_checkboxes_binning and on_binning:
                status_checkboxes_binning.append(True)
            else:
                status_checkboxes_binning.append(False)
            if i in got_status_checkboxes_transform and on_transform:
                status_checkboxes_transform.append(True)
            else:
                status_checkboxes_transform.append(False)
            if i in got_status_checkboxes_scaling and on_scaling:
                status_checkboxes_scaling.append(True)
            else:
                status_checkboxes_scaling.append(False)

        status_checkboxes_preprocessing = dict(zip(names_features, status_checkboxes_preprocessing))
        status_checkboxes_handling_outliners = dict(zip(names_features, status_checkboxes_handling_outliners))
        status_checkboxes_binning = dict(zip(names_features, status_checkboxes_binning))
        status_checkboxes_transform = dict(zip(names_features, status_checkboxes_transform))
        status_checkboxes_scaling = dict(zip(names_features, status_checkboxes_scaling))
        status_checkboxes = [False for i in range(len(names_features))]
        for number in got_status_checkboxes_features:
            status_checkboxes[number] = True
        status_checkboxes = dict(zip(names_features, status_checkboxes))
        status_radio = [False for i in range(len(names_all_features))]
        status_radio[index_target] = True
        status_radio = dict(zip(names_all_features, status_radio))

        names_for_json.append('Dataset')
        data_for_json.append((status_checkboxes, status_radio))
        names_for_json.append('Processing_missing')
        data_for_json.append(status_checkboxes_preprocessing)
        names_for_json.append('Handling_outliners')
        data_for_json.append(status_checkboxes_handling_outliners)
        names_for_json.append('Binning')
        data_for_json.append(status_checkboxes_binning)
        if on_binning:
            names_for_json.append('Number_bins')
            data_for_json.append(number_bins)
        names_for_json.append('Transform')
        data_for_json.append(status_checkboxes_transform)
        if on_transform:
            names_for_json.append('Type_transform')
            data_for_json.append(type_transform)
        names_for_json.append('Scaling')
        data_for_json.append(status_checkboxes_scaling)
        if on_scaling:
            names_for_json.append('Type_scaling')
            data_for_json.append(type_scaling)
        with open(MEDIA_ROOT + '/info_algorithm.json','w') as file:
            json.dump(prepare_for_json(names_for_json, data_for_json), file)
        if 'RUN' in request.POST:
            with open(MEDIA_ROOT + '\info_algorithm.json') as file :
                data_from_json = json.load(file)
                info_data = data_from_json['Dataset']
                class_problem = data_from_json['Class_problem']
            if THREAD[0].is_alive() :
                THREAD[0].terminate()
            else :
                if os.path.isfile(MEDIA_ROOT + '\output.txt') :
                    os.remove(MEDIA_ROOT + '\output.txt')
                THREAD[0] = Process(target=algorithm_manager, args=(MEDIA_ROOT, '\data.csv', class_problem,))
                THREAD[0].start()

        return render(request, 'myapp/processing.html',
                      {'columns_feature' : names_select_features, 'rows_feature' : select_features.to_dict('records'),
                       'column_targets' : name_targets, 'rows_targets' : targets.to_dict('records'),
                       'status_checkboxes':status_checkboxes, 'status_radio': status_radio,
                       'on_processing_missing': on_processing_missing,
                       'status_processing_missing': status_checkboxes_preprocessing,
                       'on_handling_outliners': on_handling_outliners,
                       'status_handling_outliners': status_checkboxes_handling_outliners,
                       'on_binning': on_binning,
                       'status_binning': status_checkboxes_binning,
                       'on_transform': on_transform,
                       'status_transform': status_checkboxes_transform,
                       'on_scaling': on_scaling,
                       'status_scaling': status_checkboxes_scaling})


def result(request):
    if request.method == 'GET':
        form = UploadFileForm()
        f = open(MEDIA_ROOT + '\\results.txt', 'r')
        return render(request, 'myapp/result.html', {'file': f.read().split('\n'), 'form':form})
    elif request.method == 'POST':
        if 'Pipeline 1' in request.POST:
            shutil.copyfile(MEDIA_ROOT + '\\pipeline0.pkl', MEDIA_ROOT + '\\pipeline.pkl')
        elif 'Pipeline 2' in request.POST:
            shutil.copyfile(MEDIA_ROOT + '\\pipeline1.pkl', MEDIA_ROOT + '\\pipeline.pkl')
        else:
            shutil.copyfile(MEDIA_ROOT + '\\pipeline2.pkl', MEDIA_ROOT + '\\pipeline.pkl')
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'],'data.csv')
        return HttpResponseRedirect('/prepared/')


def prepared(request):
    if request.method == 'GET':
        df = pd.read_csv(MEDIA_ROOT + '\\data.csv')
        features = df
        try :
            names_all_features = np.array(df.columns, dtype=float)
            names_features = np.array(features.columns, dtype=float)
            names_all_features = get_names(len(names_all_features))
            names_features = get_names(len(names_features))
        except ValueError :
            names_all_features = df.columns
            names_features = features.columns
        status_checkboxes = [True for _ in range(len(names_all_features))]
        status_checkboxes = dict(zip(names_all_features, status_checkboxes))
        return render(request, 'myapp/prepared.html',
                      {'columns_feature' : names_features, 'rows_feature' : features.to_dict('records'),
                       'status_checkboxes' : status_checkboxes})
    elif request.method == 'POST':
        if 'Change' in request.POST:
            df = pd.read_csv(MEDIA_ROOT + '\data.csv')
            got_status_checkboxes_features = np.array(request.POST.getlist('checks[]'), dtype=int) - 1
            print(got_status_checkboxes_features)
            all_features = df.copy()
            select_features = all_features.copy()
            for index in range(len(all_features.columns)) :
                if index not in got_status_checkboxes_features :
                    select_features = select_features.drop(all_features.columns[index], 1)
            try:
                names_all_features = np.array(df.columns, dtype=float)
                names_all_features = get_names(len(names_all_features))
                names_features = np.array(all_features.columns, dtype=float)
                names_features = get_names(len(names_features))
                names_select_features = np.array(select_features.columns, dtype=float)
                names_select_features = get_names(len(names_select_features), got_status_checkboxes_features)
            except ValueError as e :
                print(e)
                names_all_features = df.columns
                names_features = all_features.columns
                names_select_features = select_features.columns
            status_checkboxes = [False for i in range(len(names_all_features))]
            for number in got_status_checkboxes_features :
                status_checkboxes[number] = True
            status_checkboxes = dict(zip(names_all_features, status_checkboxes))
            return render(request, 'myapp/prepared.html',
                          {'columns_feature' : names_select_features,
                           'rows_feature' : select_features.to_dict('records'),
                           'status_checkboxes' : status_checkboxes})
        elif 'RUN' in request.POST or 'Download' in request.POST:
            df = pd.read_csv(MEDIA_ROOT + '\data.csv')
            # df = pd.read_csv(path + filename)
            with open(MEDIA_ROOT + '\info_algorithm.json') as file :
                info_data = json.load(file)
            features = df.copy()
            got_status_checkboxes_features = np.array(request.POST.getlist('checks[]'), dtype=int) - 1
            print(got_status_checkboxes_features)
            all_features = df.copy()
            select_features = all_features.copy()
            for index in range(len(all_features.columns)) :
                if index not in got_status_checkboxes_features :
                    select_features = select_features.drop(all_features.columns[index], 1)
            try :
                names_all_features = np.array(df.columns, dtype=float)
                names_all_features = get_names(len(names_all_features))
                names_features = np.array(all_features.columns, dtype=float)
                names_features = get_names(len(names_features))
                names_select_features = np.array(select_features.columns, dtype=float)
                names_select_features = get_names(len(names_select_features), got_status_checkboxes_features)
            except ValueError as e :
                print(e)
                names_all_features = df.columns
                names_features = all_features.columns
                names_select_features = select_features.columns
            status_checkboxes = [False for i in range(len(names_all_features))]
            for number in got_status_checkboxes_features :
                status_checkboxes[number] = True
            features = select_features
            status_checkboxes = dict(zip(names_all_features, status_checkboxes))
            preprocessor = preprocessing.PreProcessing(features, None)
            preprocessor.enc = pickle.load(open(MEDIA_ROOT + '\\preprocessor.pkl', 'rb')).enc
            preprocessor.one_hot_check()
            if np.any(np.array(list(info_data['Processing_missing'].values()))) :
                try :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if
                         info_data['Processing_missing'][features.columns[i]]])
                except Exception :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Processing_missing'][f'Feature {i}']])
                preprocessor.processing_missing_values(features=changed_features)
            if np.any(np.array(list(info_data['Handling_outliners'].values()))) :
                try :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if
                         info_data['Handling_outliners'][features.columns[i]]])
                except Exception :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Handling_outliners'][f'Feature {i}']])
                preprocessor.handling_outliners(features=changed_features)
            if np.any(np.array(list(info_data['Binning'].values()))) :
                try :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Binning'][features.columns[i]]])
                except Exception :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Binning'][f'Feature {i}']])
                preprocessor.binning(info_data['Number_bins'], features=changed_features)
            if np.any(np.array(list(info_data['Transform'].values()))) :
                try :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Transform'][features.columns[i]]])
                except Exception :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Transform'][f'Feature {i}']])
                preprocessor.transform(info_data['Type_transform'], features=changed_features)
            if np.any(np.array(list(info_data['Scaling'].values()))) :
                try :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Scaling'][features.columns[i]]])
                except Exception :
                    changed_features = np.array(
                        [i for i in range(len(features.columns)) if info_data['Scaling'][f'Feature {i}']])
                preprocessor.scaling(info_data['Type_scaling'], features=changed_features)
            preprocessor.one_hot_encoder_categorical_features()
            pipeline = pickle.load(open(MEDIA_ROOT + '\\pipeline.pkl', 'rb'))
            changed_df = preprocessor.get_dataframe()
            # features = changed_df.drop(changed_df.columns[-1], 1)
            result = pd.DataFrame(pipeline.predict(changed_df))
            if 'RUN' in request.POST:
                return render(request, 'myapp/prepared.html',
                              {'columns_feature' : names_select_features,
                               'rows_feature' : select_features.to_dict('records'),
                               'column_targets' : ['Target'], 'rows_targets' : result.to_dict('records'),
                               'status_checkboxes' : status_checkboxes})
            else:
                file_path = os.path.join(MEDIA_ROOT, 'new_data.csv')
                pd.DataFrame(pd.concat((select_features, result),axis=1)).to_csv(file_path, index=False)
                if os.path.exists(file_path) :
                    with open(file_path, 'rb') as fh :
                        response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
                        response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
                        return response
                raise Http404


def download(request):
    file_path = os.path.join(MEDIA_ROOT, 'data.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404