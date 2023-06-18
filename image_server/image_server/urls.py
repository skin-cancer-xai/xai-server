"""
URL configuration for image_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from image_app import views

urlpatterns = [
    path('doctor/predict_image', views.predict_image, name='predict_image'),
    #path('developer/', views.developer_page, name='developer_page'),
    path('doctor/feedback', views.doctor_feedback, name='doctor_feedback'),
    path('developer/feedback', views.developer_feedback, name='developer_feedback'),
    path('developer/resultList', views.result_list, name='result_list'),  # 결과 목록 조회
    path('developer/result', views.result, name='result'),    
]
