from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel

import json
import os
import time
import argparse
import torch
import random

MODEL_DIRECTORY_MAP = {
    "CodeLlama-7b" : "/raid/ganesh/nagakalyani/Downloads/CodeLlama-7b-Instruct-hf",
    "CodeStral-22b" : "/raid/ganesh/nagakalyani/nagakalyani/Tushar/codestral",
    "Gemma" : "/raid/ganesh/nagakalyani/nagakalyani/Tushar/gamma27B",
    "Qwen32b" : "/raid/ganesh/nagakalyani/Downloads/Qwen-32B"
}

def create_zero_shot_prompt(code,task):
    '''
    Creates a zero shot prompt 
    '''
        
    prompt = '''### Buggy Code : 
{}


### Task :
{}
'''.format(code, task)

    return prompt


def create_zero_shot_prompts(buggy_codes,task):
    '''
    Create zero-shot prompts for a set of codes
    '''
    zero_shot_prompts = {}
    for id in sorted(buggy_codes.keys()):
        code = buggy_codes[id]

        zero_shot_prompts[id] = create_zero_shot_prompt(code,task)

    return zero_shot_prompts

def format_user_prompt(prompt, system_prompt=""):
    """
    Formats a single input string to a Qwen compatible format.

    Args : 
        prompt (str) : The user prompt (buggy code + task)
        system_prompt (str) : The system prompt 

    Returns : 
        A prompt format compatible with Qwen
    """
    formatted_prompt = (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{prompt}[/INST]"
    )
    return formatted_prompt

def convert_to_chat_format(user_prompt, system_prompt=""):
    """
    Converts a list of user messages (strings) into the expected chat format
    for tokenizer.apply_chat_template.
    
    Parameters:
        user_prompt (List[str]): List of user inputs.
        system_prompt (str): Optional system instruction.

    Returns:
        List[Dict[str, str]]: Formatted chat messages.
    """
    chat = []

    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})

    if user_prompt:
        chat.append({"role": "user", "content": user_prompt})

    return chat


# Code-Repair Task Dataset
def create_buggy_fixed_dicts(buggy_path, fixed_path):
    buggy = {}
    fixed = {}

    with open(buggy_path, 'r', encoding='utf-8') as buggy_file, \
         open(fixed_path, 'r', encoding='utf-8') as fixed_file:
        
        for idx, (buggy_line, fixed_line) in enumerate(zip(buggy_file, fixed_file), start=1):
            buggy[idx] = buggy_line.strip()
            fixed[idx] = fixed_line.strip()
    
    return buggy, fixed

def generate_single_response(model, tokenizer, user_prompt, max_length=1024, system_prompt="", device="cuda:4"):
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

    # formatted_prompt = format_user_prompt(user_prompt, system_prompt=system_prompt)
    # Convert plain user strings into chat format
    
    messages = convert_to_chat_format(user_prompt, system_prompt)

    # Format the prompt using the tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True,
                             max_length=max_length, add_special_tokens=False).to(device)

    # Generate
    output = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.0,
        max_new_tokens=512,
        output_scores=True
    )

    # Decode response
    new_tokens = output[0][model_inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    end_time = time.time()
    return response

def grade_k_shot(model, tokenizer, system_prompt, zero_shot_prompts, output_file_path, device="cuda:0", max_length=1024):
    '''
    Does zero shot grading 
    '''
    responses = {}
    prompt_file_path = output_file_path.replace(".json", "_prompt.txt")
    for student_id in sorted(zero_shot_prompts.keys()):
        user_prompt = zero_shot_prompts[student_id]

        string_response = generate_single_response(
            model, tokenizer, user_prompt, system_prompt=system_prompt, device=device, max_length=max_length)
        
        responses[student_id] = string_response

        with open(output_file_path.replace(".json", ".txt"), "a") as f:
            f.write(f"{student_id} : ")
            f.write(string_response)
            f.write("\n")
        with open(prompt_file_path, "a") as f:
            f.write(f"{student_id} : ")
            messages = convert_to_chat_format(user_prompt, system_prompt)
            f.write(str(messages))
            f.write("\n\n")
        

    with open(output_file_path, "w") as f:
        json.dump(responses, f)

def initialize_model_and_tokenizer(model_size="32b", adapter_path="", device="cuda:4"):
    """
    Loads and returns the model and tokenizer.

    Args : 
        model_size (str) : "7b", "13b" or "34b"
        adapter_path (str) : Used when we want to load an adapter
        device (str) : Specific device where we want to load the model

    Returns : 
        A tuple of tokenizer and the model
    """
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_directory_path, torch_dtype=torch.bfloat16, device_map=device).eval()
    tokenizer.padding_side = "right"

    ## Loading adapters
    if (adapter_path != "") :
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    end_time = time.time()
    print(f"Loaded model and tokeniser in {end_time - start_time} seconds")

    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_size', type=str, default="7b",
                        help="The size of the CodeLlama to use")

    # Done for Loading Fine-Tuned Models
    parser.add_argument('--adapter_path', type=str, default="",
                        help="The path of the LoRA adapter to use")
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="The device where all tensors are to be present")
    parser.add_argument('--output_file_path', type=str,
                        help="Path for model outputs")
    parser.add_argument('--max_length', type=int, default=4096,
                        help="The max length for the tokenizer")
    parser.add_argument('--buggy_path', type=str, default="", help="Path to Buggy Test Dataset")
    parser.add_argument('--fixed_path', type=str, default="", help="Path to Fixed Test Dataset")


    torch.manual_seed(0)
    random.seed(0)

    args = parser.parse_args()

    start_time = time.time()

    # Extract system prompt
    system_prompt = "You are a helpful assistant that fixes buggy code. You will be given a buggy code snippet and are expected to return ONLY the corrected version of the code. Do not include any explanations or comments.\n"

    # Load the model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(
        model_size=args.model_size, device=args.device, adapter_path=args.adapter_path)
    
    #Task
    task="Fix the bug in the code above and return ONLY the corrected code.\n"
    
    buggy_dict,fixed_dict = create_buggy_fixed_dicts(args.buggy_path,args.fixed_path)
    zero_shot_prompts = create_zero_shot_prompts(buggy_dict,task)
    grade_k_shot(model, tokenizer, system_prompt, zero_shot_prompts,
                     args.output_file_path, device=args.device, max_length=args.max_length)

