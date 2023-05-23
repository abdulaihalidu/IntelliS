
from django.urls import path
from base.views import *
urlpatterns = [
    path('', admin_page, name="admin-page"),
    path('chat/', chat_page, name='chat'),
    path('register/', signUpPage, name="register"),
    path('login/', logInPage, name="login"),
    path('my-profile/', patient_profile, name="patient-profile"),
    path("patient-list/", PatientList.as_view(), name="patients"),
    path("disease-list/", DiseaseList.as_view(), name="diseases"),
    path('logout/', logoutUser, name='logout'),
    path('get/', model_response, name='model-response')
]
