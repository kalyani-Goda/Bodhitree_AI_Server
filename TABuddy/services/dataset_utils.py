import os
import json
import sys
import csv
import ast
import random
import pandas as pd 
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def json_from_string (string) : 
    return ast.literal_eval(string.strip())

def truncate_to_100_words(text):
    words = text.split()
    if len(words) > 100:
        return ' '.join(words[:100])
    return text


def extract_llm_ratings (lab_results_path, criterion_name, criterion_responses="") :
    predicted_results = {}

    # LLM outputss
    
    count = 0
    print(f'The criterion response is {criterion_responses}')
    for student_id, model_response in criterion_responses.items():
        stripped_model_response = model_response.strip()
        start_index = model_response.find('{')
        end_index = model_response.find('}') + 1

        content_within_braces = model_response[start_index:end_index]
        
        already_extracted = 1
        try : 
            extracted_ans = json_from_string(content_within_braces)
            print(f"\n exteacted ans is {extracted_ans} \n")
            already_extracted = 0
        except : 
            if (stripped_model_response.startswith('''{\n"answer": "''')) :
                option = stripped_model_response[13]
            elif (stripped_model_response.startswith('''{\"answer\" : ''')) :
                option = stripped_model_response[12] 
            elif (stripped_model_response.startswith("The correct answer is ")) : 
                option = stripped_model_response[22]
            elif (stripped_model_response.startswith("Answer: ")):
                option = stripped_model_response[8]
            else : 
                count += 1
                    # print(student_id, model_response)
                continue
        reasoning="I am unable to provide the reasoning for this criterion."        
        if not (already_extracted) : 
            try:
                option = extracted_ans['answer'][0]
            except Exception as e:
                continue
            try:
                reasoning = extracted_ans['reasoning']
            except Exception as e:
                continue
        try : 
            option = option.capitalize()
        except Exception as e : 
            pass
            
        diff = ord(option) - ord('A')
        if not(diff >= 0 and diff < 4) : 
            # print(student_id, model_response[:20])
            continue
        print(f'the option is {option}')
        reasoning = truncate_to_100_words(reasoning)    
        result=[]
        result.append(option)
        result.append(reasoning)
        print(result)
        predicted_results[student_id] = result
       

    return predicted_results