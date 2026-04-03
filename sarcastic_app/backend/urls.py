from django.urls import path

from . import views

urlpatterns = [
    path('',views.index),
    path('predict/',views.predict_view,name="predict"),
]