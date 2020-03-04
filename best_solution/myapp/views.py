from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm

# Imaginary function to handle an uploaded file.
from handlers.handlers_for_site import handle_uploaded_file, file_to_alg

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # handle_uploaded_file(request.FILES['file'])
            # file_to_alg('C:\\Users\\adels\PycharmProjects\project_coursework\site_cw\site_cw\media\data.csv')
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
        return render(request, 'myapp/upload.html', {'form': form})

def preprocessing(request):
    if request.method == 'GET':
        
