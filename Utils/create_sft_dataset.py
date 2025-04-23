import os
import json
import argparse

# Code-Repair Task Dataset
def convert_to_jsonl(buggy_path, fixed_path, output_path):
    system_prompt ="You are a helpful assistant that fixes buggy code. You will be given a buggy code snippet and are expected to return ONLY the corrected version of the code. Do not include any explanations or comments.\n"
    task = "Fix the bug in the code above and return ONLY the corrected code.\n"
    with open(buggy_path, 'r', encoding='utf-8') as buggy_file, \
         open(fixed_path, 'r', encoding='utf-8') as fixed_file, \
         open(output_path, 'w', encoding='utf-8') as out_file:
        
        for buggy, fixed in zip(buggy_file, fixed_file):
            buggy = buggy.strip()
            fixed = fixed.strip()

            prompt = (
                "<s>[INST] <<SYS>>\n"
                f"{system_prompt}"
                "<</SYS>>\n\n"
                f"### Buggy Code :\n{buggy}\n\n"
                f"### Task :\n{task}[/INST]"
            )
 
            # Output JSON line
            json.dump({"prompt": prompt, "completion": f"{fixed} </s>"}, out_file)
            out_file.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Convert buggy and fixed text files to JSONL format.")
    parser.add_argument('--train_path', required=True, help="Directory path for training data.")
    parser.add_argument('--test_path', required=False, help="Directory path for test data.")     
    parser.add_argument('--eval_path', required=True, help="Directory path for evaluation data.")
    parser.add_argument('--output_path', required=True, help="Common output directory for the JSONL files.")
    
    args = parser.parse_args()
    
    # Convert train and eval data to JSONL format in the common output directory
    convert_to_jsonl(
        buggy_path=f"{args.train_path}/buggy.txt",
        fixed_path=f"{args.train_path}/fixed.txt",
        output_path=f"{args.output_path}/train.jsonl"
    )

    convert_to_jsonl(
        buggy_path=f"{args.eval_path}/buggy.txt",
        fixed_path=f"{args.eval_path}/fixed.txt",
        output_path=f"{args.output_path}/eval.jsonl"
    )

if __name__ == "__main__":
    main()
