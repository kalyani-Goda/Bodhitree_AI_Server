import django
from django.core.management.base import BaseCommand
from django.db.models import Prefetch
from .models import Problem, Submission, Criteria, Rating, GradingHistory
import os
import json
import tempfile
import subprocess
import time
from celery import shared_task
from .services.dataset_create import *
from .services.retraining import *
from .services.test_inference import *
from .services.mlflow_inference import *
import subprocess
from subprocess import call
import logging as logger
from django.conf import settings  # Import settings
logger = logger.getLogger(__name__)

def run_dvc_command(command):
    """Run DVC command from project root"""
    try:
        subprocess.run(
            command,
            cwd=str(settings.BASE_DIR),  # Update with your actual project root path
            check=True,
            shell=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}, {str(e)}")
        raise

def append_jsonl_files(source_file_path, target_file_path):
    # Open the source file in read mode and target file in append mode
    with open(source_file_path, 'r') as source_file, open(target_file_path, 'a') as target_file:
        # Read from the source file line by line
        for line in source_file:
            # Append each line to the target file
            target_file.write(line)
            

def extract_data():
    """Function to extracting the data from the Database"""
    # Fetch all Problem instances 
    all_problems = Problem.objects.all()
    # Choose problem where isTrained is False
    untrained_problems = [problem for problem in all_problems if not problem.isTrained]
    # print(f"Untrained Problems: {untrained_problems}")
    totalDataPoints = 0

    # Dictionary to store the final JSON structure
    json_data = {}

    for problem in untrained_problems:
        totalDataPoints += problem.totalDataPoints
        # Fetch The Problem Statement
        try:
            problem_id = problem.id  
            problem = Problem.objects.get(id=problem_id)
            problemStatement = problem.problem_statement
        except Problem.DoesNotExist:
            print(f"No problem found with ID: {problem_id}")
            return

        # Fetch all Submissions for the given problem
        submissions = Submission.objects.filter(problem_id=problem_id)
        studentSubmissions = {submission.student_id: submission.source_code for submission in submissions}

        # Fetch the Rubrics for the given problem
        rubrics = {}
        try:
            # Fetch all criteria for the given problem, including their associated ratings
            criteria_with_ratings = Criteria.objects.filter(problem_id=problem_id).prefetch_related('rating_set')
            
            for criteria in criteria_with_ratings:
                criteriaDict = {'description': criteria.description}
                ratingsDict = {rating.title: rating.description for rating in criteria.rating_set.all()}
                if not ratingsDict:
                    print("  No ratings available for this criteria.")
                criteriaDict['ratings'] = ratingsDict
                rubrics[criteria.title] = criteriaDict
        except Problem.DoesNotExist:
            print(f"No problem found with ID: {problem_id}")
        
        # Fetch the Original Grades for the given problem
        grades = {}
        submissions = Submission.objects.filter(problem_id=problem_id)

        for submission in submissions:
            grading_histories = GradingHistory.objects.filter(submission=submission.id)
            
            for grading_history in grading_histories:
                criterion_title = grading_history.criteria.title
                student_id = submission.student_id
                try:
                    manual_marks = grading_history.manual_rating.title
                except GradingHistory.manual_rating.RelatedObjectDoesNotExist:
                    print("Manual Rating: Not present")
                    continue
                
                # Initialize the nested dictionary for each criterion title if it doesn't exist
                if criterion_title not in grades:
                    grades[criterion_title] = {}

                # Store the student_id as the key and manual_rating.marks as the value
                grades[criterion_title][student_id] = manual_marks
        # Populate the JSON dictionary for the current problem
        json_data[problem_id] = {
            'problem_statement': problemStatement,
            'student_submissions': studentSubmissions,
            'rubrics': rubrics,
            'grades': grades,
        }
    return json.dumps(json_data), untrained_problems, totalDataPoints

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=1)
def handle_retraining_process(self):
    try:
        # Fetch the MLflow tarcking model name and the previous version models storing folder path from settings file
        model_name = settings.MODEL_NAME
        download_path = settings.DOWNLOAD_PATH
        # Step 1: Extract data and calculate data points
        data, untrained_problems, total_data_points = extract_data()
        
        if total_data_points <= settings.THRESHOLD_DATA_POINTS:
            logger.error("Insufficient data points: {}".format(total_data_points))
            return "Insufficient data points"

        # Step 2: Create DPO train and test datasets
        train_path, eval_path, test_path = create_datasets(data)
        
        # Add to DVC
        run_dvc_command('dvc pull')

        # Step 3: Start retraining
        start_retraining(train_path, eval_path, test_path, model_name, download_path)
        
        # update or append the test dataset file with the new test data
        append_jsonl_files(test_path, settings.TESTDATASET_FILE_PATH)


        # Check if DVC needs to add or update files
        if subprocess.run(['dvc', 'status'], cwd=str(settings.BASE_DIR), capture_output=True, text=True).stdout:
            run_dvc_command('dvc add Datacollector/utils/test_dataset')
            run_dvc_command('dvc commit Datacollector/utils/test_dataset.dvc')
            run_dvc_command('dvc push')
        else:
            logger.info("No changes to push to DVC")
        
        # If retraining successful, update the `isTrained` status
        print("Retraining process completed successfully.")
        for problem in untrained_problems:
            problem.isTrained = True
            problem.save()
            
        # Step 4: calculate the accuracy of the model after inference on the model with the test data
        mlfolw_inference(settings.EXPERIMENT_NAME, model_name, test_path, download_path)
        
        return "Retraining process completed successfully"
    except Exception as e:
        logger.error("Error in grading process: {}".format(str(e)))
        raise self.retry(exc=e)


    