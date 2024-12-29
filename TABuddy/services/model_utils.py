from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
import json
import os
import time
import argparse
import ast

# from utils import get_submission_strings, extract_code
from peft import PeftModel
from accelerate import Accelerator

def format_user_prompt (prompt, system_prompt=""):
    """
    Formats a single input string to a CodeLlama compatible format.

    Args : 
        prompt (str) : The user prompt
        system_prompt (str) : The system prompt (Optional)

    Returns : 
        A prompt format compatible with CodeLlama
    """
    if (system_prompt) :
        formatted_prompt = f"<s>[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n\\n{prompt} [/INST]"
    else : 
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    return formatted_prompt

def generate_single_response(model, tokenizer, user_prompt, device, max_length=1024, system_prompt=""):

    """
    Generates a response for a single user prompt.

    Args : 
        model : The model which has been loaded into memory
        tokenizer : The tokenizer which has been loaded into memory
        user_prompt (str) : The user prompt
        max_length (int) : The maximum input length
        system_prompt (str) : The system prompt
        device (str) : The device on which the inference is going to run 

    Returns : 
        A string response from the model
    """
    start_time = time.time()

    formatted_prompt = format_user_prompt(user_prompt, system_prompt=system_prompt)

    # print(f"device is {device}")

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=max_length ,add_special_tokens=False).to(device)

    output = model.generate(
        **inputs,
        # attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.1,
        temperature=0.1,
        max_new_tokens=512
    )

    # Extract the new tokens (response) from the generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    end_time = time.time()

    return response
