from django.shortcuts import render
from rest_framework.views import APIView
from scripts import phishing_sites
from ai_models.utils.customeresponse import create_response


# Create your views here.

class VerificationPhishingSites(APIView):

    def get(self, request):

        url = self.request.query_params.get('url', None)
        keyword = self.request.query_params.get('keyword', None)

        try:
            data = phishing_sites.route_pipeline(url=url, keyword=keyword)

        except Exception as e:
            print(e)
            return create_response(result='tholvi aadaidhu vittai maganeaa', status=False, code=400, message="valvea maiyam endhan valvea maiyam fuck.")

        return create_response(result=data, status=True, code=200, message="vada en machi valaka bajiii")
