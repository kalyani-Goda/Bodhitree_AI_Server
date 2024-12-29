from .dataset_utils import * 
from .grade_utils import * 
from .model_utils import *
import random


def create_lora_prompt (context, code, task, options) : 
    """
    Creates prompt for SFT with LoRA

    Args : 
        context (str) : The simplified problem statement
        code (str) : The student code
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
    Returns : 
        prompt (str) : A prompt for performing SFT with LoRA
    """
    options_list = ""
    for key in sorted(options.keys()) : 
        options_list += f"{key}. {options[key]}\n"

    prompt = '''### Context : 
{}

### Code : 
{}

### Task :
{}

### Options :
{}
### Response : The required output in json format is :'''.format(context, code, task, options_list)
    return prompt   # New format in the last line

def create_dpo_prompt (context, code, task, options) : 
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
    for key in sorted(options.keys()) : 
        options_list += f"{key}. {options[key]}\n"

    prompt = '''### Context : 
{}

### Code : 
{}

### Task :
{}

### Options :
{}
### Response : The required ouptut in json format is :'''.format(context, code, task, options_list)
    return prompt   # New format in the last line

def get_lora_dataset (context, codes, task, options, original_grades,split=0.7) : 
    """
    Creates dataset for a single lab for SFT with LoRA

    Args : 
        context (str) : The simplified problem statement
        codes (dict) : A dictionary of student codes. Student ids are the keys
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
        original_grades (dict) : A dictionary of TA assigned grades. Student ids are the keys
        split (float) : The train split for the dataset
    Returns : 
        A tuple of train and test splits. Both the splits are lists
        Each item in the split is a tuple of student id, prompt and the expected response
    """
    lora_prompts = {}
    for student_id in sorted(codes.keys()) :
        if student_id not in original_grades :
            continue
        original_grade = original_grades[student_id]
        # response = f'''The correct answer is {original_grade}. {options[original_grade]}'''
        response = '''{"answer" : ''' + f'''"{original_grade}. {options[original_grade]}"''' + '''}'''  # New format
        student_code = codes[student_id]
        lora_prompts[student_id] = (create_lora_prompt(context, student_code, task, options), response)
    
    # Split the dictionary into two lists based on the split parameter
    num_items = len(lora_prompts)
    split_index = int(num_items * split)

    train_set = [(key, value[0], value[1]) for idx, (key, value) in enumerate(lora_prompts.items()) if idx < split_index]
    test_set = [(key, value[0], value[1]) for idx, (key, value) in enumerate(lora_prompts.items()) if idx >= split_index]

    return train_set,test_set

def get_dpo_dataset (context, codes, task, options, original_grades,system_prompt, split=0.7) : 
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

    for student_id in sorted(codes.keys()) :
        if student_id not in original_grades :
            continue
        original_grade = original_grades[student_id]
        # chosen_response = f'The correct answer is {original_grade}. {options[original_grade]} </s>'

        chosen_response = '''{"answer" : ''' + f'''"{original_grade}. {options[original_grade]}"''' + '''} </s>'''
        
        rejected_responses = []
        for option in options.keys() : 
            if (option != original_grade) : 
                # rejected_response = f'The correct answer is {option}. {options[option]} </s>'
                rejected_response = '''{"answer" : ''' + f'''"{option}. {options[option]}"''' + '''} </s>'''
                rejected_responses.append(rejected_response)

        student_code = codes[student_id]
        lora_prompts[student_id] = create_dpo_prompt(context, student_code, task, options)
        chosen[student_id] = chosen_response
        rejected[student_id] = rejected_responses
    
    # Split the dictionary into two lists based on the split parameter
    num_items = len(lora_prompts)
    split_idx = int(num_items * split)

    train_set = []
    test_set = []

    for idx, (key, value) in enumerate(lora_prompts.items()) : 
        prompt = format_user_prompt(lora_prompts[key], system_prompt)
        chosen_response = chosen[key]
        if (idx < split_idx) :    
            for rejected_response in rejected[key] : 
                train_set.append({"prompt" : prompt, "chosen" : chosen_response, "rejected" : rejected_response})
        else : 
            for rejected_response in rejected[key] : 
                test_set.append({"prompt" : prompt, "chosen" : chosen_response, "rejected" : rejected_response})

    return train_set,test_set


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()

    parser.add_argument("--parent_dir", type=str,
                        help="The folder where all the labs are present")
    parser.add_argument("--eval_lab_names", nargs="+",
                        help="List of the names of labs used for evaluation. All other labs will be used in the training split")
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Fraction of data to be used for training")
    
    parser.add_argument('--system_prompt_path', type=str, default="",
                        help="The file where system prompt is provided")
    parser.add_argument('--lora', type=int, default=1,
                        help="Create dataset for lora or dpo")
    parser.add_argument('--shuffle', type=int, default=0,
                        help="Whether to shuffle the samples in the dataset")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Whether to use the verbose problem statement")

    parser.add_argument('--train_dataset_path', type=str, default="",
                        help="Path to the train dataset")
    parser.add_argument('--test_dataset_path', type=str, default="",
                        help="Path to the evaluation dataset")

    args = parser.parse_args()


    # Extract system prompt
    with open(args.system_prompt_path, "r") as f :
        system_prompt = f.read().strip()

    train_dataset_file = open(args.train_dataset_path, "w")
    test_dataset_file = open(args.test_dataset_path, "w")

    train_points = []
    test_points = []

    for lab in os.listdir(args.parent_dir) : 
        lab_path = os.path.join(args.parent_dir, lab)

        train_split = args.train_split
        lab_grades_path = os.path.join(lab_path, "rubric_ratings.csv")

        if (os.path.isdir(lab_path)) :
            if (lab in args.eval_lab_names) : 
                train_split = 0             ## The entire lab will be used for evaluation if it's name is in args.eval_lab_names

            # Extract context (Simplified Problem Statement)
            if (args.verbose) :
                context_path = os.path.join(lab_path, "ps.txt")
            else : 
                context_path = os.path.join(lab_path, "modified_ps.txt")
            with open(context_path, "r") as f : 
                context = f.read().strip()

            # Dict of student ids and their submissions
            student_submissions = get_submission_json(lab_path)

            # Get the criterion and rating descriptions
            all_criteria = get_rubrics(lab_path)

            # Get the original TA grades for that lab
            original_grades = extract_all_original_grades(lab_grades_path)

            # Repeat for all criteria
            for criterion in all_criteria.keys() : 
                criterion_desc = all_criteria[criterion]["description"]  # Description for that particular criterion
                options = all_criteria[criterion]["ratings"]             # Rating descriptions for that particular criterion

                #Get the Grades And Rubrics related to specific criterion
                criterion_original_grades = original_grades[criterion]

                ## For json output
                # task = f'''Choose the option which is most suitable for the above code for the criterion "{criterion_desc}". Give your output as a json with a single field "answer". Do not output anything else. Strictly follow this output format.'''
                
                ## For sentence output
                task = f'''Choose the option which is most suitable for the above code for the criterion "{criterion_desc}". Your response must start with "The correct answer is ". Strictly follow this output format at any cost'''
                
                if (args.lora) : 
                    train_user_prompt_set, test_user_prompt_set = get_lora_dataset(context, student_submissions, task, options, 
                                                                                    criterion_original_grades, split=train_split)

                    train_prompt_set = list(map(lambda x : format_prompt_and_response(x[1], x[2], system_prompt=system_prompt), train_user_prompt_set))
                    test_prompt_set = list(map(lambda x : format_prompt_and_response(x[1], x[2], system_prompt=system_prompt), test_user_prompt_set))

                    for train_prompt in train_prompt_set :
                        json_obj = {}
                        json_obj["text"] = train_prompt 
                        train_points.append(json_obj)

                    for test_prompt in test_prompt_set : 
                        json_obj = {}
                        json_obj["text"] = test_prompt 
                        test_points.append(json_obj)
                else : 
                    ## New trial
                    train_set, test_set = get_dpo_dataset(context, student_submissions, task, options, 
                                                            criterion_original_grades, system_prompt, split=train_split)          
                    
                    for train_data_point in train_set :
                        train_points.append(train_data_point)

                    for test_data_point in test_set : 
                        test_points.append(test_data_point)
                        

    if (args.shuffle) : 
        random.shuffle(train_points)
        random.shuffle(test_points)
    
    for train_data_point in train_points : 
        json.dump(train_data_point, train_dataset_file)
        train_dataset_file.write('\n')

    for test_data_point in test_points :
        json.dump(test_data_point, test_dataset_file)
        test_dataset_file.write('\n')

    train_dataset_file.close()
    test_dataset_file.close()