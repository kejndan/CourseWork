from django.http import HttpResponseRedirect
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
        status_checkboxes = [True for _ in range(len(names_features))]
        status_checkboxes = dict(zip(names_features, status_checkboxes))
        status_radio = [False for _ in range(len(names_all_features)-1)] + [True]
        status_radio = dict(zip(names_all_features, status_radio))
        with open(MEDIA_ROOT + '/info_algorithm.json', 'w') as file:
            json.dump(prepare_for_json(status_checkboxes, status_radio), file)
        return render(request, 'myapp/processing.html',
                      {'columns_feature': names_features, 'rows_feature': features.to_dict('records'),
                       'column_targets': name_targets, 'rows_targets': targets.to_dict('records'),
                       'status_checkboxes': status_checkboxes, 'status_radio': status_radio})
    elif request.method == 'POST':
        names_for_json = []
        data_for_json = []
        names_for_json.append('Class_problem')
        data_for_json.append(request.POST.get("type_func"))
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')
        index_target = int(request.POST.get('checksradio[]')) - 1
        all_features = df.drop(df.columns[index_target], 1)
        targets= df[df.columns[index_target:index_target+1]]
        got_status_checkboxes = np.array(request.POST.getlist('checks[]'), dtype=int) - 1
        if np.where(got_status_checkboxes == index_target):
            np.delete(got_status_checkboxes, np.where(got_status_checkboxes == index_target))
        select_features = all_features.copy()
        for index in range(len(all_features.columns)):
            if index not in got_status_checkboxes:
                select_features = select_features.drop(all_features.columns[index], 1)
        try:
            names_all_features = np.array(df.columns, dtype=float)
            names_all_features = get_names(len(names_all_features))
            names_features = np.array(all_features.columns, dtype=float)
            names_features = get_names(len(names_features))
            names_select_features = np.array(select_features.columns, dtype=float)
            names_select_features = get_names(len(names_select_features), got_status_checkboxes)
            name_targets = ['Target']
        except ValueError as e:
            print(e)
            names_all_features = df.columns
            names_features = all_features.columns
            names_select_features = select_features.columns
            name_targets = targets.columns

        status_checkboxes = [False for i in range(len(names_features))]
        for number in got_status_checkboxes:
            status_checkboxes[number] = True
        status_checkboxes = dict(zip(names_features, status_checkboxes))
        status_radio = [False for i in range(len(names_all_features))]
        status_radio[index_target] = True
        status_radio = dict(zip(names_all_features, status_radio))
        names_for_json.append('Dataset')
        data_for_json.append((status_checkboxes, status_radio))
        with open(MEDIA_ROOT + '/info_algorithm.json','w') as file:
            json.dump(prepare_for_json(names_for_json, data_for_json), file)
        return render(request, 'myapp/processing.html',
                      {'columns_feature' : names_select_features, 'rows_feature' : select_features.to_dict('records'),
                       'column_targets' : name_targets, 'rows_targets' : targets.to_dict('records'),
                       'status_checkboxes':status_checkboxes, 'status_radio': status_radio})

def working(request):
    if request.method == 'POST':
        with open(MEDIA_ROOT + '\info_algorithm.json') as file :
            data_from_json = json.load(file)
            info_data = data_from_json['Dataset']
            class_problem = data_from_json['Class_problem']
        if THREAD[0].is_alive():
            THREAD[0].terminate()
        else:
            if os.path.isfile(MEDIA_ROOT+'\output.txt') :
                os.remove(MEDIA_ROOT+'\output.txt')
            THREAD[0] = Process(target=algorithm_manager, args=(MEDIA_ROOT, '\data.csv', class_problem,))
            THREAD[0].start()
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')

        features = df.copy()
        for number, name in enumerate(info_data.keys()) :
            if not info_data[name] :
                features = features.drop(df.columns[number], 1)
            if info_data[name] == 'Target' :
                features = features.drop(df.columns[number], 1)
                target = df[df.columns[number :number + 1]]
        try:
            names_all_features = np.array(df.columns, dtype=float)
            names_features = np.array(features.columns, dtype=float)
            names_all_features = get_names(len(names_all_features))
            names_features = get_names(len(names_features))
            name_targets = ['Target']
        except ValueError:
            names_all_features = df.columns
            names_features = features.columns
            name_targets = target.columns
        status_checkboxes = [True for i in range(len(names_features))]
        status_checkboxes = dict(zip(names_features, status_checkboxes))
        status_radio = [False for i in range(len(names_all_features)-1)] + [True]
        status_radio = dict(zip(names_all_features, status_radio))
        return render(request, 'myapp/processing.html',
                      {'columns_feature' : names_features, 'rows_feature' : features.to_dict('records'),
                       'column_targets' : name_targets, 'rows_targets' : target.to_dict('records'),
                       'status_checkboxes' : status_checkboxes, 'status_radio' : status_radio})


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
        features = df.drop(df.columns[-1], 1)
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
            got_status_checkboxes = np.array(request.POST.getlist('checks[]'), dtype=int) - 1
            print(got_status_checkboxes)
            all_features = df.copy()
            select_features = all_features.copy()
            for index in range(len(all_features.columns)) :
                if index not in got_status_checkboxes :
                    select_features = select_features.drop(all_features.columns[index], 1)
            try:
                names_all_features = np.array(df.columns, dtype=float)
                names_all_features = get_names(len(names_all_features))
                names_features = np.array(all_features.columns, dtype=float)
                names_features = get_names(len(names_features))
                names_select_features = np.array(select_features.columns, dtype=float)
                names_select_features = get_names(len(names_select_features), got_status_checkboxes)
            except ValueError as e :
                print(e)
                names_all_features = df.columns
                names_features = all_features.columns
                names_select_features = select_features.columns
            status_checkboxes = [False for i in range(len(names_all_features))]
            for number in got_status_checkboxes :
                status_checkboxes[number] = True
            status_checkboxes = dict(zip(names_all_features, status_checkboxes))
            return render(request, 'myapp/prepared.html',
                          {'columns_feature' : names_select_features,
                           'rows_feature' : select_features.to_dict('records'),
                           'status_checkboxes' : status_checkboxes})
        elif 'RUN' in request.POST:
            df = pd.read_csv(MEDIA_ROOT + '\data.csv')
            got_status_checkboxes = np.array(request.POST.getlist('checks[]'), dtype=int) - 1
            print(got_status_checkboxes)
            all_features = df.copy()
            select_features = all_features.copy()
            for index in range(len(all_features.columns)) :
                if index not in got_status_checkboxes :
                    select_features = select_features.drop(all_features.columns[index], 1)
            try :
                names_all_features = np.array(df.columns, dtype=float)
                names_all_features = get_names(len(names_all_features))
                names_features = np.array(all_features.columns, dtype=float)
                names_features = get_names(len(names_features))
                names_select_features = np.array(select_features.columns, dtype=float)
                names_select_features = get_names(len(names_select_features), got_status_checkboxes)
            except ValueError as e :
                print(e)
                names_all_features = df.columns
                names_features = all_features.columns
                names_select_features = select_features.columns
            status_checkboxes = [False for i in range(len(names_all_features))]
            for number in got_status_checkboxes :
                status_checkboxes[number] = True
            status_checkboxes = dict(zip(names_all_features, status_checkboxes))
            pipeline = pickle.load(open(MEDIA_ROOT + '\\pipeline.pkl', 'rb'))
            print(select_features)
            result = pd.DataFrame(pipeline.predict(select_features))
            return render(request, 'myapp/prepared.html',
                          {'columns_feature' : names_select_features,
                           'rows_feature' : select_features.to_dict('records'),
                           'column_targets' : 'Target', 'rows_targets' : result.to_dict('records'),
                           'status_checkboxes' : status_checkboxes})









