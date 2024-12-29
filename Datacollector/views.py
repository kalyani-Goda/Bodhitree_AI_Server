from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from .models import Problem, Submission, Criteria, Rating, GradingHistory
from django.utils import timezone
from rest_framework import status
from django.db import transaction
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import json

def home(request):
    return HttpResponse("Welcome to Data Collector")

def dummy(id):
    reverse_id = id[::-1]
    return reverse_id

class DataPoint(APIView):
    """
    An API view to handle GET and POST requests.
    """

    def get(self, request, format=None):
        # Write Logic for handling GET requests
        data = {'message': 'GET request received'}
        return Response(data, status=status.HTTP_200_OK)
    
    def post(self, request, format=None):
        try:    
            data = request.data # Extract the data from the request
            with open('data.txt', 'w') as f:
                f.write(str(data))
            user_id = data['user_id'] # User ID of person sending the data
            data = data['0']
            problems_already_present = []

            for lab_name in data.keys():
                lab_id = int(lab_name) # Lab ID
                for program_activity in data[lab_name].keys():    
                    #Extract Program Activity ID
                    program_id = int(program_activity)

                    program_data = data[lab_name][program_activity]

                    #Extract Problem Statement    
                    problem_statement = data[lab_name][program_activity]['problem_statement']
                    
                    filename = data[lab_name][program_activity]['file_name'] # Filename
                    
                   #Create Problem Object
                    problem, created = Problem.objects.get_or_create(
                        id=program_id,
                        defaults={
                            'problem_statement': problem_statement,
                            'user_id': user_id,
                            'lab_id': lab_id
                        }
                    )
                    if not created:
                        print(f"Problem with ID {program_id} already exists")
                    else:
                        print("Problem objects created successfully")
                    
                    # Initialize grading history counter for this problem
                    grading_history_count = 0

                    # Create Criteria Object 
                    criteria=data[lab_name][program_activity]['rubric']
                    for criterion in criteria:
                        for criterion_id, criterion_details in criterion.items():
                            criterion_title = criterion_details['title']
                            criterion_description = criterion_details['description']
                            criteria_object, created = Criteria.objects.get_or_create(
                                id = int(criterion_id),
                                problem = problem,
                                defaults = {
                                    'title': criterion_title,
                                    'description': criterion_description
                                }
                            )
                            if not created:
                                print(f"Criteria with ID {criterion_id} already exists")
                            else:
                                print("Criteria objects created successfully")

                            #Create Rating Details Object First and associate with Criteria
                            for rating_id,rating_data in criterion_details['ratings'].items(): 
                                rating_title = rating_data['title']
                                rating_description = rating_data['description']
                                rating_marks = rating_data.get('marks', 0)  # Use default 0 if not provided
                                try:
                                    rating_object, created=Rating.objects.get_or_create(
                                        id = int(rating_id),
                                        criteria = criteria_object,
                                        defaults = {
                                            'title': rating_title,
                                            'description': rating_description,
                                            'marks': int(rating_marks),  
                                        }
                                    )
                                except Exception as e:
                                    print(f"Error: {str(e)}")
                                if not created:
                                    print(f"Rating with ID {rating_id} already exists")
                                else:
                                    print("Rating objects created successfully")
                                
                    for student_id in program_data['student_submissions'].keys():
                        student_id = student_id # Student ID
                        dummy_id = dummy(student_id) # Dummy ID
                        student_submission_details = data[lab_name][program_activity]['student_submissions'][student_id]
                        source_code = student_submission_details['source_code'] # Source Code
                        
                        # Create Submission object
                        submission,created = Submission.objects.get_or_create(
                            filename = filename,
                            student_id = student_id,
                            dummy_id = dummy_id,
                            source_code = source_code,
                            problem = problem
                        )
                        if not created:
                            print(f"Submission with ID {submission.id} already exists")
                        else:
                            print("Submission object created successfully")
                        
                        for criterion_id in student_submission_details['manual_rating'].keys():
                            # Fetch the Criteria object based on program_id and id
                            fetch_criteria = Criteria.objects.filter(problem = program_id, id = int(criterion_id)).first()

                            if fetch_criteria:
                                # Extract manual rating and comments from the student_submission_details
                                manual_rating_id, manual_comments = student_submission_details['manual_rating'].get(criterion_id, (None, ''))
                                manual_rating_id = int(manual_rating_id) if manual_rating_id != 'None' else -1

                                # Extract AI rating and comments from the student_submission_details
                                ai_rating_id, ai_comments = student_submission_details['ai_rating'].get(criterion_id, (None, ''))
                                ai_rating_id = int(ai_rating_id) if ai_rating_id != 'None' and ai_rating_id is not None else -1

                                # Fetch the manual and ai rating object based on program_id and id
                                fetch_manual_rating = Rating.objects.filter(id = int(manual_rating_id)).first()
                                fetch_ai_rating = Rating.objects.filter(id = int(ai_rating_id)).first()

                                # Check if a GradingHistory object with the same problem_id, student_id, and criteria exists
                                existing_entry = GradingHistory.objects.filter(
                                    submission=submission,
                                    criteria=fetch_criteria  # Using the Criteria instance directly
                                )
                                
                                if not existing_entry:
                                    # Create a new GradingHistory object 
                                    GradingHistory.objects.create(
                                        submission=submission,
                                        criteria=fetch_criteria,  # Using the Criteria instance directly
                                        manual_rating=fetch_manual_rating,
                                        ai_rating=fetch_ai_rating,
                                        manual_comments=manual_comments,
                                        ai_comments=ai_comments
                                    )
                                    grading_history_count += 1
                                else:
                                    print(f"Entry already exists for program_id={program_id}, student_id={student_id}, criterion_id={criterion_id}")
                            else:
                                print(f"Criterion with ID {criterion_id} not found for program_id={program_id}")
                    problem.totalDataPoints += grading_history_count
                    problem.save()
            return Response({"message": "Data received"}, status=status.HTTP_200_OK)
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return Response({"error": "An error occurred"}, status=status.HTTP_400_BAD_REQUEST)

def data_collector_metrics(request):
    return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)