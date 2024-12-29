import json
import os
import random
import time
import torch
# from .model_utils import *
from .grade_utils import *

def grade_submissions(
    tokenizer, model, device, problem_statement, submissions, criterion_info, criterion_name="", max_length=4096, few_shot=False, few_shot_examples=0, train_split=0.7
):
    torch.manual_seed(0)
    random.seed(0)

    start_time = time.time()

    # Extract student submissions
    student_submissions = submissions
    student_submissions_copy = {}
    sorted_student_ids = sorted(student_submissions.keys())
    for i in range(len(student_submissions)):
        student_id = sorted_student_ids[i]
        student_submissions_copy[student_id] = student_submissions[student_id]
    student_submissions = student_submissions_copy

    # Extract system prompt
    system_prompt = '''Your task is to choose the MOST suitable option among a set of options I provide, about a code which will also be provided. Give your output as a json with a single field "answer". Do not output anything else. Strictly follow this output format at any cost.'''

    # Extract the context (Simplified problem statement)
    context = problem_statement

    # Extract criterions
    criterion_descs = []
    criterion_ids = []
    options_list = []

    for criterion_obj in criterion_info:
        # print(f"the criteria obj is {criterion_obj}")
        options = {}
        criterion_id = criterion_obj["id"]
        criterion_desc = criterion_obj["description"]
        raw_options = criterion_obj["Ratings"]

        for json_obj in raw_options:
            options[json_obj["title"]] = json_obj["description"]

        criterion_ids.append(criterion_id)
        criterion_descs.append(criterion_desc)
        options_list.append(options)


    outputs = {}
    for idx in range(len(criterion_descs)):
        criterion_desc = criterion_descs[idx]
        criterion_id = str(criterion_ids[idx])
        options = options_list[idx]
        if not few_shot:
            # This describes the task for the LLM
            task = f'''Choose the option which is most suitable for the above code for the criterion "{criterion_desc}". Give your output as a json with two fields : "answer" and "reasoning". Do not output anything else. Strictly follow this output format.'''

            zero_shot_prompts = create_zero_shot_prompts(context, student_submissions, task, options)
            response = grade_k_shot(model, tokenizer, system_prompt, zero_shot_prompts, device, max_length=max_length)
            outputs[criterion_id] = response
    

    end_time = time.time()
    print("Total time taken :", end_time - start_time)
    return outputs
