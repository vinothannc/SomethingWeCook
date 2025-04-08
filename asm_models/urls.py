from django.urls import path, include
from asm_models import views as asm_models_views

urlpatterns = [

    path('verification_sites/', asm_models_views.VerificationPhishingSites.as_view())

]
