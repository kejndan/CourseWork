from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
# Imaginary function to handle an uploaded file.
from handlers.handlers_for_site import handle_uploaded_file, file_to_alg
from best_solution.settings import MEDIA_ROOT

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
        data = pd.read_csv(MEDIA_ROOT+'\data.csv')
        return render(request, 'myapp/processing.html', {'columns': data.columns, 'rows': data.to_dict('records')})


