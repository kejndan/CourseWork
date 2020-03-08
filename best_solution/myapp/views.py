from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
# Imaginary function to handle an uploaded file.
from .tasks import handle_uploaded_file, file_to_alg
from best_solution.settings import MEDIA_ROOT
from threading import Thread


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            # file_to_alg('C:\\Users\\adels\PycharmProjects\project_coursework\site_cw\site_cw\media\data.csv')
            return HttpResponseRedirect('/processing/')
    else:
        form = UploadFileForm()
        return render(request, 'myapp/upload.html', {'form': form})

def processing(request):
    if request.method == 'GET':
        df = pd.read_csv(MEDIA_ROOT+'\data.csv')
        # file_to_alg(MEDIA_ROOT + '\data.csv')
        Thread(target=file_to_alg, args=(MEDIA_ROOT + '\data.csv',)).start()
        # print(res.get(timeout=10.0))
        return render(request, 'myapp/processing.html',
                      {'columns' : df.columns, 'rows' : df.to_dict('records')})

def update(request):
    if request.method == 'GET':
        if request.is_ajax():
            pass


