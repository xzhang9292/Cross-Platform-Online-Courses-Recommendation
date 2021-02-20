from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('home', views.home, name='blog-home'),
    path('about/', views.about, name='blog-about'),
    path('', views.search,name='search'),
]
