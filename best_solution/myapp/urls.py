from django.urls import path

from . import views

urlpatterns = [
    path('index/', views.upload_file, name='index'),
    path('preprocessing/', )
]