from django.urls import path, re_path
from django.conf.urls.static import static
from best_solution.settings import MEDIA_ROOT, MEDIA_URL
from . import views

urlpatterns = [
    path('', views.upload_file, name='index'),
    path('processing/', views.processing),
    path('processing/working', views.working),
    path('processing/result', views.result),
    path('prepared/', views.prepared)
]


urlpatterns += static(MEDIA_URL, document_root=MEDIA_ROOT)