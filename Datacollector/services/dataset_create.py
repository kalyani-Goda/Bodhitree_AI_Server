import os
import json
import csv
import django
from django.conf import settings


def format_user_prompt(prompt, system_prompt=""):
    """
    Formats a single input string to a CodeLlama compatible format
    """
    if system_prompt:
        formatted_prompt = (
            f"<s>[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n\\n{prompt} [/INST]"
        )
    else:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    return formatted_prompt


def process_reasoning(lab_path):
    grades_file_path = None

    # Check if the directory exists and find the CSV file
    if os.path.isdir(lab_path):
        for file in os.listdir(lab_path):
            if file == "rubric_ratings.csv":
                grades_file_path = os.path.join(lab_path, file)
                break

    if not grades_file_path:
        raise FileNotFoundError("rubric_ratings.csv not found in the provided lab_path")

    # Process the CSV file
    with open(grades_file_path, "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    crit_ratings = {}
    for i in range(len(rows[0])):
        if i == 0:
            continue
        if rows[0][i] not in crit_ratings:
            crit_ratings[rows[0][i]] = {}
            crit_ratings[rows[0][i]][rows[2][i]] = rows[3][i]
        else:
            crit_ratings[rows[0][i]][rows[2][i]] = rows[3][i]

    student_reasoning2 = {}
    for j in range(len(rows)):
        if j <= 5:
            continue
        id = rows[j][0].split("@")[0]
        for i in range(len(rows[0])):
            if i == 0:
                continue
            if rows[0][i] not in student_reasoning2:
                student_reasoning2[rows[0][i]] = {}
            if id not in student_reasoning2[rows[0][i]]:
                student_reasoning2[rows[0][i]][id] = {}
            if rows[j][i] == "0" or rows[j][i] == "No Comments" or rows[j][i] == "1":
                student_reasoning2[rows[0][i]][id][rows[2][i]] = crit_ratings[
                    rows[0][i]
                ][rows[2][i]]
            else:
                student_reasoning2[rows[0][i]][id][rows[2][i]] = rows[j][i]

    return student_reasoning2

def create_dpo_prompt(context, code, task, options):
    """
    Creates prompt for finetuning with DPO

    Args :
        context (str) : The simplified problem statement
        code (str) : The student code
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
    Returns :
        prompt (str) : A prompt for finetuning with DPO
    """
    options_list = ""
    for key in sorted(options.keys()):
        options_list += f"{key}. {options[key]}\n"

    prompt = """### Context : 
{}

### Code : 
{}

### Task :
{}

### Options :
{}
### Response : The required output in json format is :""".format(
        context, code, task, options_list
    )
    return prompt  # New format in the last line

def get_dpo_dataset(
    context, codes, task, options, original_grades, system_prompt, split=0.7
):
    """
    Creates dataset for a single lab for finetuning with DPO

    Args :
        context (str) : The simplified problem statement
        codes (dict) : A dictionary of student codes. Student ids are the keys
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
        original_grades (dict) : A dictionary of TA assigned grades. Student ids are the keys
        system_prompt (str) : The system prompt
        split (float) : The train split for the dataset
    Returns :
        A tuple of train and test splits. Both the splits are lists
        Each item in the split is a json with three fields "prompt", "chosen" and "rejected"
    """
    lora_prompts = {}
    chosen = {}
    rejected = {}

    for student_id in sorted(codes.keys()):
        if student_id not in original_grades:
            continue
        original_grade = original_grades[student_id]
        # chosen_response = f'The correct answer is {original_grade}. {options[original_grade]} </s>'
        original_grade = original_grade.strip()
        chosen_response = (
            """{"answer" : """
            + f'''"{original_grade}. {options[original_grade]}"'''
            + """} </s>"""
        )

        rejected_responses = []
        for option in options.keys():
            if option != original_grade:
                # rejected_response = f'The correct answer is {option}. {options[option]} </s>'
                rejected_response = (
                    """{"answer" : """
                    + f'''"{option}. {options[option]}"'''
                    + """} </s>"""
                )
                rejected_responses.append(rejected_response)

        student_code = codes[student_id]
        lora_prompts[student_id] = create_dpo_prompt(
            context, student_code, task, options
        )
        chosen[student_id] = chosen_response
        rejected[student_id] = rejected_responses

    # Split the dictionary into two lists based on the split parameter
    num_items = len(lora_prompts)
    split_idx = int(num_items * split)

    train_set = []
    test_set = {}
    prompts = {}
    target_grades = {}

    for idx, (key, value) in enumerate(lora_prompts.items()):
        prompt = format_user_prompt(lora_prompts[key], system_prompt)
        chosen_response = chosen[key]
        if idx < split_idx:
            for rejected_response in rejected[key]:
                train_set.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    }
                )
        else:
            prompts[key] = prompt
            target_grades[key] = original_grades[key]
    test_set["prompts"] = prompts
    test_set["target_grades"] = target_grades
    return train_set, test_set


def get_dpo_reasoning_dataset(
    context,
    codes,
    task,
    options,
    original_grades,
    original_reasonings,
    system_prompt,
    split=0.7,
):
    """
    Creates dataset for a single lab for finetuning with DPO

    Args :
        context (str) : The simplified problem statement
        codes (dict) : A dictionary of student codes. Student ids are the keys
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
        original_grades (dict) : A dictionary of TA assigned grades. Student ids are the keys
        system_prompt (str) : The system prompt
        split (float) : The train split for the dataset
    Returns :
        A tuple of train and test splits. Both the splits are lists
        Each item in the split is a json with three fields "prompt", "chosen" and "rejected"
    """
    lora_prompts = {}
    chosen = {}
    rejected = {}

    for student_id in sorted(codes.keys()):
        if student_id not in original_grades:
            continue
        original_grade = original_grades[student_id]
        original_reasoning = original_reasonings[student_id]
        # chosen_response = f'The correct answer is {original_grade}. {options[original_grade]} </s>'

        chosen_response = (
            """{"answer" : """
            + f'''"{original_grade}. {options[original_grade]} , "reasoning" : {original_reasoning}"'''
            + """} </s>"""
        )

        rejected_responses = []
        for option, rating in options.items():
            if option != original_grade:
                # rejected_response = f'The correct answer is {option}. {options[option]} </s>'
                rejected_response = (
                    """{"answer" : """
                    + f'''"{option}. {options[option]} , "reasoning" : {rating}"'''
                    + """} </s>"""
                )
                rejected_responses.append(rejected_response)

        student_code = codes[student_id]
        lora_prompts[student_id] = create_dpo_prompt(
            context, student_code, task, options
        )
        chosen[student_id] = chosen_response
        rejected[student_id] = rejected_responses

    # Split the dictionary into two lists based on the split parameter
    num_items = len(lora_prompts)
    split_idx = int(num_items * split)

    train_set = []
    test_set = []

    for idx, (key, value) in enumerate(lora_prompts.items()):
        prompt = format_user_prompt(lora_prompts[key], system_prompt)
        chosen_response = chosen[key]
        if idx < split_idx:
            for rejected_response in rejected[key]:
                train_set.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    }
                )
        else:
            for rejected_response in rejected[key]:
                test_set.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    }
                )

    return train_set, test_set


def create_datasets(json_data):

    # with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.loads(json_data)

    # system_prompt_file = "/home/iitb_admin_user/kalyani/Bodhitree_AI_Server/Datacollector/utils/prompts/dpo_sys_prompt.txt"
    system_prompt_file = settings.SYSTEM_PROMPT
    
    # task_file = "/home/iitb_admin_user/kalyani/Bodhitree_AI_Server/Datacollector/utils/prompts/task.txt"
    task_file = settings.TASK_PROMPT

    # Extract system prompt
    with open(system_prompt_file, "r") as f:
        system_prompt = f.read().strip()
    
    with open(task_file, "r") as f:
        task = f.read().strip()

    # train_path = "/home/iitb_admin_user/kalyani/Bodhitree_AI_Server/Datacollector/utils/Retraining_datasets/train.jsonl"
    # test_path = "/home/iitb_admin_user/kalyani/Bodhitree_AI_Server/Datacollector/utils/Retraining_datasets/test.jsonl"
    # eval_path = "/home/iitb_admin_user/kalyani/Bodhitree_AI_Server/Datacollector/utils/Retraining_datasets/eval.jsonl"

    train_path = settings.TRAIN_PATH
    test_path = settings.TEST_PATH
    eval_path = settings.EVAL_PATH
    train_dataset_file = open(train_path, "w")
    test_dataset_file = open(test_path, "w")
    eval_dataset_file = open(eval_path, "w")

    train_points = []
    test_points = {}

    for labid,lab in data.items():
        train_split = 0.8
        context = lab['problem_statement']
            
        # Dict of student ids and their submissions
        student_submissions = lab['student_submissions']

        # Get the criterion and rating descriptions
        all_criteria = lab['rubrics']
        
        # Get the original TA grades for that lab
        original_grades = lab['grades']
        if(len(original_grades)==0) :
            continue
        #original_reasonings = process_reasoning(lab_path)
        # Repeat for all criteria
        test_points[labid] = {}
        for criterion in all_criteria.keys():
            # Description for that particular criterion
            criterion_desc = all_criteria[criterion]["description"]
            # Rating descriptions for that particular criterion
            options = all_criteria[criterion]["ratings"]
            # Get the Grades And Rubrics related to specific criterion
            criterion_original_grades = original_grades[criterion]

            # Format the string by replacing `{}` with `criterion_desc`
            llm_task = task.format(criterion_desc)

        
            train_set, test_set = get_dpo_dataset(
                context,
                student_submissions,
                llm_task,
                options,
                criterion_original_grades,
                system_prompt,
                split=train_split,
            )

            for train_data_point in train_set:
                train_points.append(train_data_point)

            test_set["ratings"] = list(options.keys())
            test_points[labid][criterion] = test_set
            

    for train_data_point in train_points[:-50]:
        json.dump(train_data_point, train_dataset_file)
        train_dataset_file.write("\n")

    for train_data_point in train_points[-50:]:
        json.dump(train_data_point, eval_dataset_file)
        eval_dataset_file.write("\n")
    print(test_points)

    for key, value in test_points.items():
        # Construct a new dictionary with only one key-value pair for each line
        line_data = {key: value}
        # Dump the dictionary as a JSON object and write it as a new line
        test_dataset_file.write(json.dumps(line_data) + '\n')


    train_dataset_file.close()
    test_dataset_file.close()
    eval_dataset_file.close()

    return train_path, eval_path, test_path
