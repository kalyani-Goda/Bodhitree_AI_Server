# In llmpredictor/views.py
import os
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from celery.result import AsyncResult
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import default_storage
import json
import time
from .serializers import FileUploadSerializer
import zipfile
from .tasks import query_codellama
from .services.dataset_utils import extract_llm_ratings
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from django.conf import settings


def home(request):
    return HttpResponse("Welcome to Auto Grading Server")

class Predictor(APIView):
    def post(self, request):
        start_time = time.time()
        if request.method == 'POST':
            data = request.data
            # save the submissions
            submissions = {}
            for key, file_list in request.FILES.lists():
                # Read the content of the file (assuming each key has a single file)
                file_content = file_list[0].read().strip()
                submissions[key] = file_content

            # Save problem statement
            ps = data['problem_statement']

            # Save criteria
            criteria_json = request.data.get('criteria')
            criteria = json.loads(criteria_json)

            try:
                task = query_codellama.apply_async(
                    args=[submissions, ps, criteria],
                    kwargs={
                            'criterion_name': "",
                            'max_length': 4096, 
                            'few_shot': False, 
                            'few_shot_examples': 0, 
                            'train_split': 0.7
                            }
                        )

                end_time = time.time()
                print("Total time taken :", end_time - start_time)
                return Response({'status code': 202, 'task': task.id}, status=202)
            except Exception as e:
                print(e)
                end_time = time.time()
                return Response(status=404, data={'status': 404, 'message': 'Something went wrong'})

    def get(self, request):
        start_time = time.time()
        
        # Access task_id from the query parameters in the GET request
        task_id = request.query_params.get('task_id')

        if not task_id:
            return Response({"error": "task_id is required"}, status=400)
    
        task_result = AsyncResult(task_id)
        if task_result.status == "SUCCESS":
            end_time = time.time()
            print("Total time taken :", end_time - start_time)
            return Response(task_result.result, status=200)
        else:
            return Response({"message": "task pending"}, status=202)

def inference_metrics(request):
    return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)