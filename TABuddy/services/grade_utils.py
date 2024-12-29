from .model_utils import *

from .dataset_utils import *
    
def create_zero_shot_prompt (context, code, task, options) : 
    """
    Creates a zero shot user prompt.

    Args : 
        context (str) : The simplified problem statement
        code (str) : The student code
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
    Returns : 
        A zero shot user prompt as a string
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

    # prompt += '''{"answer" : '''
    return prompt


def create_zero_shot_prompts (context, codes, task, options) :
    """
    Create zero-shot prompts for a set of student submissions.
        
    Args : 
        context (str) : Modified problem statement
        codes (dict) : A dictionary of all the student codes. Keys are the student ids
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")

    Returns : 
        A dictionary of zero-shot user prompts for all the students. Keys are the student ids
    """
    zero_shot_prompts = {}
    for student_id in sorted(codes.keys()) : 
        student_code = codes[student_id]

        zero_shot_prompts[student_id] = create_zero_shot_prompt(context, student_code, task, options)

    return zero_shot_prompts


def grade_k_shot (model, tokenizer, system_prompt, zero_shot_prompts, device, max_length=1024, text_dump=False) :

    '''
    Grades student submissions using zero-shot and few-shot prompting.

    Args : 
        model : The model which has been loaded into memory
        tokenizer : The tokenizer which has been loaded into memory
        system_prompt (str) : The system prompt which will be used for grading all submissions
        zero_shot_prompts (dict) : A dictionary of user prompts (0-shot or few-shot). Student ids are the keys
        output_file_path (str) : The path to the file to print all the model responses
        device (str) : The device where the grading will be done
        max_length (int) : The maximum input length. Rest of the input will be truncated

    Returns : 
        None 
    ''' 
    responses = {}

    for student_id in sorted(zero_shot_prompts.keys()) :
        user_prompt = zero_shot_prompts[student_id]
        
        string_response = generate_single_response(model, tokenizer, user_prompt, device, system_prompt=system_prompt, max_length=max_length)
        
        responses[student_id] = string_response
        
    return responses