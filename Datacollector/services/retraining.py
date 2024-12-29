import os
import django
from django.conf import settings

# def setup_django():
#     os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bodhitree_AI_Server.settings')
#     django.setup()
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from trl import DPOTrainer, DPOConfig
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset
from accelerate import Accelerator
import torch
from typing import Dict, Optional
from dataclasses import dataclass, field
import time
from datetime import datetime
import wandb
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
import pandas as pd
from mlflow import MlflowClient
# Call setup_django before importing settings
# setup_django()



# Explicitly set credentials and endpoint URL in the script
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MLFLOW_S3_ENDPOINT_URL  # Adjust this as necessary

# Set the MLflow tracking URI
mlflow.set_tracking_uri(settings.TRACKING_URI)

mlflow.set_experiment(settings.EXPERIMENT_NAME)

# MODEL_DIRECTORY_MAP = {
#     "7b" : "/home/iitb_admin_user/kalyani/Bodhitree_AI_Server/base_models/CodeLlama-7b-Instruct-hf"
# }

MODEL_DIRECTORY_MAP = settings.MODEL_DIRECTORY_MAP

def setup_mlflow_experiment(experiment_name):
    """
    Sets up MLflow experiment, creating it if it doesn't exist
    """
    try:
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new MLflow experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            print(f"Using existing MLflow experiment '{experiment_name}' with ID: {experiment.experiment_id}")
            
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        print(f"Error setting up MLflow experiment: {str(e)}")
        raise


def download_model_artifacts(model_name, download_path):
    """
    Attempts to download model artifacts if they exist, returns None for first-time model creation
    """
    client = MlflowClient()
    
    try:
        # Check if the model exists
        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.RestException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                print(f"No existing model found with name '{model_name}'. This will be the first version.")
                return "None"
            raise e

        # If model exists, get latest version
        try:
            # Using search_model_versions instead of deprecated get_latest_versions
            model_versions = client.search_model_versions(f"name='{model_name}'")
            if not model_versions:
                print(f"Model '{model_name}' exists but has no versions. This will be the first version.")
                return "None"
                
            # Sort versions by version number and get the latest
            latest_version = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
            
            # Get the run ID associated with this model version
            run_id = latest_version.run_id
            version = latest_version.version
            
            # Define the path where you want to save the artifacts
            local_dir = os.path.join(download_path, f"{model_name}_v{version}")
            os.makedirs(local_dir, exist_ok=True)
            
            # Download the "model/checkpoints" folder artifact
            try:
                client.download_artifacts(run_id, "model/checkpoints", local_dir)
                adapter_dir = os.path.join(local_dir, "model/checkpoints")
                print(f"Successfully downloaded model artifacts. Adapter directory: {adapter_dir}")
                return adapter_dir
            except Exception as e:
                print(f"Error downloading artifacts: {str(e)}")
                return "None"
                
        except Exception as e:
            print(f"Error accessing model versions: {str(e)}")
            return "None"
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return "None"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():

        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def initialize_model_and_tokenizer_dpo(
    model_size="7b", adapter_path="", device="cuda:0", quantization_config=None
):
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_size]
    tokenizer = AutoTokenizer.from_pretrained(
        model_directory_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_directory_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": Accelerator().local_process_index},
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_directory_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    # Loading adapters
    if adapter_path != "":
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    end_time = time.time()
    print(f"Loaded model and tokeniser in {end_time - start_time} seconds")

    return tokenizer, model



def start_retraining(train_path, eval_path, test_path, model_name, download_path):
    
    # First set up the MLflow experiment
    setup_mlflow_experiment(settings.EXPERIMENT_NAME)


    # Download model artifacts that is the latest model if they exist from mlflow client
    adapter_path = download_model_artifacts(model_name, download_path)

    # Start a new MLflow run
    with mlflow.start_run(run_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
        wandb_project = "Bodhitree_AI_retraining"
        lora_r = 16
        lora_alpha = 16
        lora_dropout = 0.1
        model_size = "7b"
        device = torch.device("cuda:0")
        output_dir = settings.OUTPUT_PATH
        train_dataset_path = train_path
        eval_dataset_path = eval_path
        test_dataset_path = test_path
        torch.manual_seed(0)

        # Load and validate JSONL datasets
        try:
            train_dataset = load_dataset(
                "json", data_files=train_dataset_path, split="train"
            )
            eval_dataset = load_dataset(
                "json", data_files=eval_dataset_path, split="train"
            )
            test_dataset = load_dataset(
                "json", data_files=test_dataset_path
            )


            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize wandb right after argument parsing
            wandb.init(project=wandb_project, 
                    name=f"CodeLlama_{current_time}",
                    config={
                        "model_size": model_size,
                        "lora_r": lora_r,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout,
                    })
            # Log configuration parameters
            mlflow.log_params({
                "model_size": model_size,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "quantization" :"True"
            })

            # Log run tags
            mlflow.set_tags({
                "model_type": "CodeLlama",
                "training_type": "DPO",
                "framework": "transformers",
                "data_format": "jsonl"
            })

            # Log datasets with JSONL-specific metadata
            for path, ds, context in [
                (train_path, train_dataset, "train_set"),
                (eval_path, eval_dataset, "evaluation_set"),
                (test_path, test_dataset, "testing_set")
            ]:
            
                # Load JSONL as a Pandas DataFrame
                df = pd.read_json(path, lines=True)
                dataset = mlflow.data.from_pandas(df, source=path)
                mlflow.log_input(dataset, context=context)
                mlflow.log_metrics({f"NO. of {context} examples": len(ds)})
                # Log the actual JSONL file
                mlflow.log_artifact(path, artifact_path="datasets")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                device_map={"": 0},  # Explicitly set GPU 4
            )

            if adapter_path == "None":

                tokenizer, model = initialize_model_and_tokenizer_dpo(
                    model_size=model_size, device=device, quantization_config=bnb_config
                )
            else:

                tokenizer, model = initialize_model_and_tokenizer_dpo(
                    model_size=model_size, adapter_path=adapter_path, device=device, quantization_config=bnb_config
                )

            model.config.use_cache = False

            
            # Log the number of trainable parameters
            training_args = DPOConfig(
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                num_train_epochs=1,
                logging_steps=1,
                save_steps=250,
                gradient_accumulation_steps=16,
                gradient_checkpointing=True,
                learning_rate=1e-5,
                # evaluation_strategy="steps",
                eval_strategy="steps",
                eval_steps=0.5,
                output_dir=output_dir,
                report_to="wandb",
                lr_scheduler_type="cosine",
                warmup_steps=100,
                optim="paged_adamw_32bit",
                fp16=True,
                remove_unused_columns=False,
                gradient_checkpointing_kwargs=dict(use_reentrant=False),
                seed=0,
                beta=0.1,
            )
            
            # lora config parameters
            peft_config = LoraConfig(
                r= lora_r,
                lora_alpha= lora_alpha,
                lora_dropout= lora_dropout,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "out_proj",
                    "fc_in",
                    "fc_out",
                    "wte",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("Model Type:", type(model))
            print("Has Generate Method:", hasattr(model, 'generate'))

            # Initialize the DPOTrainer
            dpo_trainer = DPOTrainer(
                model,
                ref_model=None,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
            )

            # Log configuration parameters into MLflow experiment
            mlflow.log_params({
                "base_model": MODEL_DIRECTORY_MAP[model_size],
                "max_prompt_length":2048,
                "max_length":1536,
                "per_device_train_batch_size":1,
                "per_device_eval_batch_size":1,
                "num_train_epochs":1,
            })

            # 6. start the training
            train_result = dpo_trainer.train()

            # Log training metrics
            mlflow.log_metrics({
                "final_loss": train_result.training_loss,
                "total_steps": train_result.global_step
            })

            # output_dir is the base directory where to save the retrianed models
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
            timestamped_dir = os.path.join(output_dir, f"model_{current_time}")

            # Create the directory if it doesn't exist
            os.makedirs(timestamped_dir, exist_ok=True)

            # Save the model inside the timestamped directory
            dpo_trainer.save_model(timestamped_dir)

            # Create a pipeline from the saved model
            saved_tokenizer = AutoTokenizer.from_pretrained(timestamped_dir)
            saved_model = AutoModelForCausalLM.from_pretrained(
                timestamped_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map = "auto"
            )
            
            # Create a text-generation pipeline
            model_pipeline = pipeline(
                "text-generation",
                model=saved_model,
                tokenizer=saved_tokenizer,
            )
            
            # Now log the pipeline with MLflow
            mlflow.transformers.log_model(
                transformers_model=model_pipeline,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Set up MLflow client for additional operations
            client = MlflowClient()
            
            # Get the latest version
            latest_versions = client.search_model_versions(f"name='{model_name}'")

            if not latest_versions:
                version_number = 1
            else:
                version_number = max(int(v.version) for v in latest_versions)
            # Convert to integer explicitly
            latest_version = int(version_number)
            
            try:
            # Wait briefly for model registration to complete
                time.sleep(2)
                
                # Set model version tags
                client.set_model_version_tag(
                    name=model_name,
                    version=str(latest_version),  # Convert to string for tag setting
                    key="training_date",
                    value=datetime.now().strftime("%Y-%m-%d")
                )
                client.set_registered_model_tag(model_name, "base_model", "CodeLlama-7b")

            except Exception as e:
                print(f"Warning: Issue with model versioning or tagging: {str(e)}")
            
            # Log the entire model directory to MLflow
            mlflow.log_artifacts(timestamped_dir, "model")

            print(f"Model saved in: {timestamped_dir}")

            # 7. save
            checkpoint_dir = os.path.join(timestamped_dir, "final_checkpoint")
            dpo_trainer.model.save_pretrained(checkpoint_dir)

            # Log the checkpoint directory into mlflow as an artifact
            mlflow.log_artifacts(checkpoint_dir, "model/checkpoints")

            print(f"Training completed. Model saved in MLflow under run: {mlflow.active_run().info.run_id}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            mlflow.log_param("training_error", str(e))
            raise

        # Finish the MLflow run
        mlflow.end_run()  

    return checkpoint_dir