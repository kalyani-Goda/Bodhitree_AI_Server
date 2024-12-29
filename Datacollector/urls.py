from django.urls import path
from . import views

app_name='Datacollector'
urlpatterns = [
    path('',views.home,name='home'),
    path('data-point/', views.DataPoint.as_view(), name='data_point'),
    path('data_collector_metrics/', views.data_collector_metrics, name='data_collector_metrics'),
]