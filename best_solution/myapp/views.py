from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.urls import reverse
from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
# Imaginary function to handle an uploaded file.
from handlers.handlers_for_site import handle_uploaded_file, remove_folder_contents,get_names
from best_solution.settings import MEDIA_ROOT, THREAD
from threading import Thread
from multiprocessing import Process
from handlers.core import algorithm_manager
import os
import numpy as np
from django.core import serializers



def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            remove_folder_contents(MEDIA_ROOT)
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/processing/')
    else:
        form = UploadFileForm()
        return render(request, 'myapp/upload.html', {'form': form})


def processing(request):
    if request.method == 'GET':
        df = pd.read_csv(MEDIA_ROOT+'\data.csv')
        features = df.drop(df.columns[-1], 1)
        targets= df[df.columns[-1:]]
        try:
            names_features = np.array(features.columns, dtype=float)
            names_features = get_names(len(names_features))
            name_targets = ['Target']
        except ValueError:
            names_features = features.columns
            name_targets = targets.columns
        return render(request, 'myapp/processing.html',
                      {'columns_feature': names_features, 'rows_feature': features.to_dict('records'),
                       'column_targets': name_targets, 'rows_targets': targets.to_dict('records')})
    elif request.method == 'POST':
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')
        return render(request, 'myapp/processing.html',
                      {'columns' : df.columns, 'rows' : df.to_dict('records')})

def working(request):
    if request.method == 'POST':
        print(request.POST.get("type_func"))
        if THREAD[0].is_alive():
            THREAD[0].terminate()
            # THREAD[0] = Process(target=file_to_alg, args=(MEDIA_ROOT,'\data.csv',))
        else:
            if os.path.isfile(MEDIA_ROOT+'\output.txt') :
                os.remove(MEDIA_ROOT+'\output.txt')
            THREAD[0] = Process(target=algorithm_manager, args=(MEDIA_ROOT, '\data.csv',request.POST.get("type_func"),))
            THREAD[0].start()
        # proc = Thread(target=file_to_alg, args=(MEDIA_ROOT + '\data.csv',)).start()
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')
        # return None
        return render(request, 'myapp/processing.html',
                      {'columns' : df.columns, 'rows' : df.to_dict('records')})

def ajax_request(request):
    f = open(MEDIA_ROOT + '\\results.txt', 'r')
    return render(request, 'myapp/result.html', {'file': f.read().split('\n')})


