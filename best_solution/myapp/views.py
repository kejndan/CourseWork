from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
# Imaginary function to handle an uploaded file.
from handlers.handlers_for_site import file_to_alg, handle_uploaded_file, remove_folder_contents
from best_solution.settings import MEDIA_ROOT
from threading import Thread




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
        return render(request, 'myapp/processing.html',
                      {'columns' : df.columns, 'rows' : df.to_dict('records')})
    elif request.method == 'POST':
        proc = Thread(target=file_to_alg, args=(MEDIA_ROOT + '\data.csv',)).start()
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')
        # return None
        return render(request, 'myapp/processing.html',
                      {'columns' : df.columns, 'rows' : df.to_dict('records')})

def working(request):
    if request.method == 'POST':
        proc = Thread(target=file_to_alg, args=(MEDIA_ROOT + '\data.csv',)).start()
        df = pd.read_csv(MEDIA_ROOT + '\data.csv')
        # return None
        return render(request, 'myapp/processing.html',
                      {'columns' : df.columns, 'rows' : df.to_dict('records')})
def update(request):
    if request.method == 'GET':
        if request.is_ajax():
            pass


