import os
from celery import shared_task
from .services.grading_bt import *
# from bitsandbytes import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from accelerate import init_empty_weights
import torch
from django.conf import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger object
logger = logging.getLogger(__name__)


def initialize_model_and_tokenizer_quantized(device="cuda:0", quantization_config=None):
    """
    Initializes and returns the tokenizer and model for inference.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"Invalid device specified: {device}")

    start_time = time.time()
    adapter_path = settings.ADAPTER_PATH
    model_directory_path = settings.MODEL_DIRECTORY_PATH

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_directory_path,
        quantization_config=quantization_config,
        device_map={"": device}
    ).eval()

    # Load adapters if available
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    end_time = time.time()
    print(f"Loaded model and tokenizer in {end_time - start_time} seconds")

    return tokenizer, model

@shared_task(bind=True)
def query_codellama(self, submissions, problem_statement, criterion_info,
                    criterion_name="", max_length=4096, few_shot=False, few_shot_examples=0, train_split=0.7):
    # Ensure the model and tokenizer are initialized
    print("Initiating auto evaluation")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    device=settings.DEVICE
    
    tokenizer, model = initialize_model_and_tokenizer_quantized(
        device=device,
        quantization_config=bnb_config
    )

    if tokenizer is None or model is None:
        raise ValueError("Tokenizer or model not initialized.")
    
    # Perform grading (your logic here)
    try:
    
        grades = grade_submissions(
            tokenizer=tokenizer, model=model, device=device, problem_statement=problem_statement,
            submissions=submissions, criterion_info=criterion_info, criterion_name=criterion_name,
            max_length=max_length, few_shot=few_shot, few_shot_examples=few_shot_examples, train_split=train_split
        )

        rating_id_map = {}

        for criterion_obj in criterion_info: 
            raw_options = criterion_obj["Ratings"]
            criterion_id = str(criterion_obj["id"])

            rating_id_map[criterion_id] = {}

            for option_obj in raw_options:
                rating_id_map[criterion_id][option_obj["title"]] = option_obj["id"]
        
        combined_json = {}
        print(f'the grades are after the model evalution: {grades}')
        for criterion_id, response in grades.items(): 
            llm_ratings = extract_llm_ratings("", "", response)
            for student_id, result in llm_ratings.items(): 
                rating = result[0]
                reasoning = result[1]
                
                llm_ratings[student_id][0] = rating_id_map[criterion_id][rating]
                llm_ratings[student_id][1] = reasoning 
                
            combined_json[criterion_id] = llm_ratings

        return combined_json

    finally:
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info("Cleaned up GPU memory.")
