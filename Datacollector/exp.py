from services.dataset_create import *
from services.retraining import *
from services.test_inference import *
from services.mlflow_inference import *
import json
import logging as logger
import subprocess
import django
from django.conf import settings

# def setup_django():
#     os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bodhitree_AI_Server.settings')
#     django.setup()

# setup_django()
model_name = settings.MODEL_NAME
download_path = settings.DOWNLOAD_PATH
experiment_name = settings.EXPERIMENT_NAME

# def run_dvc_command(command):
#     """Helper function to run a DVC command"""
#     try:
#         subprocess.run(command, check=True, shell=True)
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Command failed: {command}, {str(e)}")
#         raise

def run_dvc_command(command):
    """Run DVC command from project root"""
    try:
        subprocess.run(
            command,
            cwd='/home/ubuntu/Bodhitree_AI_Server',  # Update with your actual project root path
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

data_file_path = settings.DATA_FILE_PATH
testdataset_file_path = settings.TESTDATASET_FILE_PATH
# Initialize an empty dictionary to store the data
data = {}

with open(data_file_path, 'r', encoding='utf-8') as json_file:
    for line in json_file:
        # Parse each line as a JSON object
        item = json.loads(line.strip())
        
        # Add the parsed item to the dictionary, assuming each line has a unique key
        # For example, if each item has a single top-level key like "cs101a23_lq01_d_q3"
        # with a nested dictionary as its value, we can use that top-level key
        for key, value in item.items():
            data[key] = value
# print(data)
print("starts the retraing with the raw data\n")

json_data = json.dumps(data)

# Add to DVC
run_dvc_command('dvc pull')

print("creating the datasets for the training testing and evaluation\n")

train_path, eval_path, test_path = create_datasets(json_data)
print(test_path)

print("fintuning the model with the new data\n")

finetuned_model_path = start_retraining(train_path, eval_path, test_path, model_name, download_path)

append_jsonl_files(test_path, testdataset_file_path)

# run_dvc_command('dvc add utils/test_dataset')
# run_dvc_command('dvc commit utils/test_dataset.dvc')
# run_dvc_command('dvc push')

# run_dvc_command('dvc add Datacollector/utils/test_dataset')
# run_dvc_command('dvc commit Datacollector/utils/test_dataset.dvc')
# run_dvc_command('dvc push')

# Check if DVC needs to add or update files
if subprocess.run(['dvc', 'status'], cwd='/home/ubuntu/Bodhitree_AI_Server', capture_output=True, text=True).stdout:
    run_dvc_command('dvc add Datacollector/utils/test_dataset')
    run_dvc_command('dvc commit Datacollector/utils/test_dataset.dvc')
    run_dvc_command('dvc push')
else:
    logger.info("No changes to push to DVC")


# # # print("doing the test inference for calculating the model accuracy\n")

# # metrics = inference(testdataset_file_path, finetuned_model_path)

# model_name = "CodeLlama-DPO"
# experiment_name = "Bodhitree_AI_retraining" 
download_path = "/home/ubuntu/Bodhitree_AI_Server/Datacollector/utils/Finetuned_models"
test_path = "/home/ubuntu/Bodhitree_AI_Server/Datacollector/utils/Retraining_datasets/test.jsonl"

mlfolw_inference(experiment_name, model_name, test_path, download_path)

print("the accuracies of the new model\n")

print(metrics)
