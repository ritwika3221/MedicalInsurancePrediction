from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('',views.FrontPage,name='frontpage'),
    path('graph/',views.GraphPage,name='graphpage'),
    path('data/',views.DataPage,name='datapage'),
    path('predict/',views.PredictPage,name='predictpage'),
    path('amount/',views.PredictPage,name='amountpage'),
    path('contact/',views.ContactPage,name='contactpage'),
    
]
