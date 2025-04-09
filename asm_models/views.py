from django.shortcuts import render
from rest_framework.views import APIView
from scripts import phishing_sites, parked_domains
from ai_models.utils.customeresponse import create_response


# Create your views here.

class PhishingSites(APIView):

    def get(self, request):

        url = self.request.query_params.get('url', None)
        keyword = self.request.query_params.get('keyword', None)

        try:
            data = phishing_sites.route_pipeline(url=url, keyword=keyword)

        except Exception as e:
            print(e)
            return create_response(result=[], status=False, code=400, message="An error occurred")

        return create_response(result=data, status=True, code=200, message="Successfully got the results")


class ParkingSites(APIView):

    def get(self, request):

        image_url = self.request.query_params.get('image_url', None)

        try:
            data = parked_domains.detect_parked_domain(image_url=image_url)

        except Exception as e:
            print(e)
            return create_response(result=[], status=False, code=400, message="An error occurred")

        return create_response(result=data, status=True, code=200, message="Successfully got the results")
