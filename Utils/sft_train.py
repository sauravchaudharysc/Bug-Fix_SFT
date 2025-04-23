from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
import argparse
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset
from accelerate import Accelerator
import torch
from typing import Dict, Optional
import time
import os
import torch
from trl import SFTTrainer, SFTConfig



MODEL_DIRECTORY_MAP = {
    "Qwen32b" : "/raid/ganesh/nagakalyani/Downloads/Qwen-32B",
}

def initialize_model_and_tokenizer_sft(
    model_size="32b", adapter_path="", device="cuda:1", quantization_config=None
):
    start_time = time.time()
    model_directory_path = MODEL_DIRECTORY_MAP[model_size]

    tokenizer = AutoTokenizer.from_pretrained(model_directory_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_directory_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": Accelerator().local_process_index},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_directory_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")

    return tokenizer, model


def tokenize_function(examples):
    # Check if 'prompt' is in the expected format
    if 'prompt' not in examples:
        raise KeyError(f"Expected 'prompt' key in examples but found keys: {list(examples.keys())}")
    
    # Print an example to help debug
    if 'prompt' in examples and len(examples['prompt']) > 0:
        print(f"Example prompt type: {type(examples['prompt'][0])}")
        print(f"Example prompt: {examples['prompt'][0][:100]}...")  # Print just the beginning
    
    # Process the inputs
    try:
        # Ensure prompts are strings
        texts = examples['prompt']
        
        # If prompts are already lists, we might need to join them
        if any(isinstance(text, list) for text in texts):
            texts = [" ".join(text) if isinstance(text, list) else text for text in texts]
        
        outputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors=None,  # This is important - let the dataset handle tensor conversion
        )
        return outputs
    except Exception as e:
        print(f"Error in tokenize_function: {e}")
        print(f"Example that caused error: {examples['prompt'][0]}")
        raise
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size", type=str, default="32b")
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--output_dir", type=str, help="The folder where the fine-tuned model is saved")
    parser.add_argument("--train_dataset_path", type=str, help="Path to the train dataset file")
    parser.add_argument("--eval_dataset_path", type=str, help="Path to the eval dataset file")

    args = parser.parse_args()

    torch.manual_seed(0)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer, model = initialize_model_and_tokenizer_sft(
        model_size=args.model_size, device=args.device, quantization_config=bnb_config
    )

   
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.config.use_cache = False

    datset_start_time = time.time()

    train_dataset = load_dataset("json", data_files=args.train_dataset_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset_path, split="train")

    # train_dataset = train_dataset.map(tokenize_function, batched=True)
    # eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    # print(train_dataset[0])
    dataset_end_time = time.time()

    training_start_time = time.time()

    # Training configuration using SFTConfig
    training_args = SFTConfig(
        packing=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=250,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=0.5,
        output_dir=args.output_dir,
        report_to="tensorboard",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        optim="paged_adamw_32bit",
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        seed=0,
    )

    # LoRA Configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    # Initialize the SFT Trainer
    sft_trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=1024,
        
    )

    # Start training
    sft_trainer.train()

    # Save the trained model with a timestamped directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(args.output_dir, f"model_{current_time}")
    os.makedirs(timestamped_dir, exist_ok=True)

    sft_trainer.save_model(timestamped_dir)
    print(f"Model saved in: {timestamped_dir}")

    checkpoint_dir = os.path.join(timestamped_dir, "final_checkpoint")
    sft_trainer.model.save_pretrained(checkpoint_dir)

    training_end_time = time.time()
    data_loading_time = dataset_end_time - datset_start_time
    training_time = training_end_time - training_start_time

    time_file_path = os.path.join(checkpoint_dir, "time.txt")
    with open(time_file_path, "w") as time_file:
        time_file.write(f"Data loading time: {data_loading_time:.2f} seconds\n")
        time_file.write(f"Training time: {training_time:.2f} seconds\n")