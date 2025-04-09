from django.urls import path, include
from asm_models import views as asm_models_views

urlpatterns = [

    path('phishing_sites/', asm_models_views.PhishingSites.as_view()),
    path('parking_sites/', asm_models_views.ParkingSites.as_view())

]
