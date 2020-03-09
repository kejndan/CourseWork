from django.urls import path

from . import views

urlpatterns = [
    path('', views.upload_file, name='index'),
    path('processing/', views.processing)
]
from django.conf.urls.static import static
from best_solution.settings import MEDIA_ROOT, MEDIA_URL

urlpatterns += static(MEDIA_URL, document_root=MEDIA_ROOT)