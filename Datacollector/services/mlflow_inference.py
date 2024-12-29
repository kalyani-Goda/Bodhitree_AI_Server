from .metrics import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import torch
import random
import time
import json
import ast
import shutil
from django.conf import settings  # Import settings
import mlflow
from mlflow.tracking import MlflowClient
torch.manual_seed(0)
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MODEL_DIRECTORY_MAP = {
#     "7b": "/home/iitb_admin_user/kalyani/base_models/CodeLlama-7b-Instruct-hf",
# }

MODEL_DIRECTORY_MAP = settings.MODEL_DIRECTORY_MAP

def json_from_string(string):
    return ast.literal_eval(string.strip())

def move_folder(source_folder,destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Copy each file from the source to the destination
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Copy the file to the destination
        shutil.copy(source_file, destination_file)
        print(f"Copied {filename} to {destination_folder}")


def initialize_model_and_tokenizer(model_size="7b", adapter_path="", device="cuda:0"):
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_directory_path, torch_dtype=torch.bfloat16, device_map=device
    ).to(device).eval()
    tokenizer.padding_side = "right"

    # Loading adapters
    if adapter_path != "":
        model = PeftModel.from_pretrained(model, adapter_path)
        model.to(device).eval()

    end_time = time.time()
    print(f"Loaded model and tokeniser in {end_time - start_time} seconds")

    return tokenizer, model

def generate_single_response(
    model, tokenizer, grading_prompt, max_length=1024, device="cuda:0"
):
    start_time = time.time()

    inputs = tokenizer(
        grading_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).to(device)

    output = model.generate(
        **inputs,
        # attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.1,
        temperature=0.1,
        max_new_tokens=512,
    )

    # Extract the new tokens (response) from the generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    # print(new_tokens)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    end_time = time.time()

    # print(f"Time taken : {end_time - start_time}seconds")
    print(f"the response:\n\n{response}")

    return response

def extract_llm_ratings(model_response):
    
    count = 0
    stripped_model_response = model_response.strip()
    start_index = model_response.find('{')
    end_index = model_response.find('}') + 1

    content_within_braces = model_response[start_index:end_index]

    already_extracted = 1
    try:
        extracted_ans = json_from_string(content_within_braces)
        already_extracted = 0
    except:
        if (stripped_model_response.startswith('''{\n"answer": "''')):
            option = stripped_model_response[13]
        elif (stripped_model_response.startswith('''{\"answer\" : ''')):
            option = stripped_model_response[12]
        elif (stripped_model_response.startswith("The correct answer is ")):
            option = stripped_model_response[22]
        elif (stripped_model_response.startswith("Answer: ")):
            option = stripped_model_response[8]
        else:
            count += 1
            # print(student_id, model_response)

    if not (already_extracted):
        try:
            option = extracted_ans['answer'][0]
        except Exception as e:
            option = "A"
            pass

    try:
        option = option.capitalize()
    except Exception as e:
        option = "A"
        pass

    diff = ord(option) - ord('A')
    if not (diff >= 0 and diff < 4):
        option = "A"
        # print(student_id, model_response[:20])
        pass
    
    return option

def get_successful_model_versions_from_experiment(model_name, experiment_id):
    client = MlflowClient()
    
    # Get all runs with status "FINISHED" for the specified experiment
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string="status = 'FINISHED'")
    
    run_ids = {run.info.run_id for run in runs}
    model_versions = []

    # Retrieve all versions of the specified model
    all_versions = client.search_model_versions(f"name='{model_name}'")

    # Filter model versions based on successful runs
    for version in all_versions:
        # Ensure the version is associated with a "FINISHED" run
        if version.run_id in run_ids:
            model_versions.append(version)
    
    return model_versions


def download_model_artifacts(model_name, version, download_path):
    client = MlflowClient()
    # Get model version info
    model_version = client.get_model_version(name=model_name, version=version)
    
    # Get the run ID associated with this model version
    run_id = model_version.run_id
    
    # Define the path where you want to save the artifacts
    local_dir = os.path.join(download_path, f"{model_name}_v{version}")
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the "model" folder artifact to the specified local directory
    client.download_artifacts(run_id, "model/checkpoints", local_dir)
    
    print(f"Downloaded model version {version} to {local_dir}")
    return local_dir

def update_metrics_for_version(model_name, version_info, metrics):
    client = MlflowClient()
    
    # Log each metric individually for the specific model version
    for metric_name, value in metrics.items():
        client.log_metric(run_id=version_info.run_id, key=metric_name, value=value)
    
    print(f"Updated metrics for {model_name} version {version_info}: {metrics}")

def promote_to_challenger(model_name, best_version):
    client = MlflowClient()

    vr = client.get_model_version(model_name, best_version)
    client.set_registered_model_alias(vr.name, f"Champion", vr.version)
    
    # Promote the model version to "Challenger" by tagging it
    client.set_model_version_tag(
        name=model_name,
        version=best_version,
        key="alias",
        value="Challenger"
    )
    print(f"Model version {best_version} for {model_name} promoted to 'Champion'")
    

def evaluate_and_promote_best_model_for_experiment(model_name, experiment_name, test_data_path, download_path):
    client = MlflowClient()

    production_model_path = settings.PRODUCTION_MODEL_PATH
    
    best_accuracy = -1
    best_version = None

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    # Retrieve all model versions associated with the specific experiment
    model_versions = get_successful_model_versions_from_experiment(model_name, experiment_id)

    for version_info in model_versions:
        version = version_info.version
        
        # Download model artifacts
        local_model_path = download_model_artifacts(model_name, version, download_path)

        adapter_dir = os.path.join(local_model_path, "model/checkpoints")
        
        # Calculate accuracy
        metrics = inference(test_data_path, adapter_dir)
       
        print(f"Model Version: {version}, Accuracy_metrics: {metrics}")
        
        # Update accuracy metric in MLflow for this version
        update_metrics_for_version(model_name, version_info, metrics)

        accuracy =  metrics["Micro overall Accuracy"]
        
        # Track the best accuracy and version
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_version = version_info
            best_model_path = local_model_path
    
    # Promote the best model version to "Challenger"
    if best_version:
        promote_to_challenger(model_name, best_version.version)
    else:
        print("No model versions found or no best version determined.")

    best_model_path = os.path.join(best_model_path, "model/checkpoints")
    move_folder(best_model_path, production_model_path)
    
    # Delete all downloaded models except the best one
    for version_info in model_versions:
        version_path = os.path.join(download_path, f"{model_name}_v{version_info.version}")
        
        shutil.rmtree(version_path)
        print(f"Deleted model version {version_info.version} at path: {version_path}")
    
    return production_model_path

def inference(test_file_path, adapter_path):
    # test_file_path = "/home/iitb_admin_user/kalyani/dataset/test7.jsonl"
    model_size = "7b"
    device = "cuda:0"
    # adapter_path = "/home/iitb_admin_user/kalyani/finetuned_models/checkpoint-4"
    max_length = 4096
    
    
    start_time = time.time()

    # Load the model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(
        model_size=model_size, device=device, adapter_path=adapter_path
    )

    # Initialize an empty dictionary to store the data
    test_data = {}

    # Read the .jsonl file line by line
    with open(test_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # Parse each line as a JSON object
            line_data = json.loads(line.strip())
            # Update the main dictionary with the parsed line
            test_data.update(line_data)

    # Now `data_dict` contains all key-value pairs from the .jsonl file
    # print(test_data)
    model_responses = {}
    labs = list(test_data.keys())
    print(labs)
    print(list(test_data[labs[0]].keys()))
    for lab_id, lab in test_data.items():
        model_responses[lab_id] = {}
        for criterion_title, details in lab.items():
            model_responses[lab_id][criterion_title] = {}
            prompts = details["prompts"]
            predictions = {}
            for student_id in sorted(prompts.keys()):
                grading_prompt = prompts[student_id]
                string_response = generate_single_response(
                        model,
                        tokenizer,
                        grading_prompt,
                        max_length,
                        device,
                    )
                response = extract_llm_ratings(string_response)
                predictions[student_id] = response
            model_responses[lab_id][criterion_title]["predictions"] = predictions
            model_responses[lab_id][criterion_title]["original_grades"] = details["target_grades"]
            model_responses[lab_id][criterion_title]["ratings"] = details["ratings"]
            
    # print(model_responses)
    metrics = dpo_results_unseen_common_macro_micro(model_responses)

    end_time = time.time()
    print("Total time taken :", end_time - start_time)
    return metrics


def mlfolw_inference(experiment_name, model_name, test_file_path, download_path):
   
    model_path = evaluate_and_promote_best_model_for_experiment(model_name, experiment_name, test_file_path, download_path)
    return model_path