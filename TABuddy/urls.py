from django.urls import path
from . import views

app_name='TABuddy'
urlpatterns = [
    path('',views.home,name='home'),
    path('tabuddy/',views.Predictor.as_view(), name='tabuddy'),
    path('inference_metrics/', views.inference_metrics, name='inference_metrics'), 
]