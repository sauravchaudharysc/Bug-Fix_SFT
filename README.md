# 🛠️ Code Repair with Supervised Fine-Tuning on CodeXGLUE

This repository presents an experiment on **supervised fine-tuning** of a large language model for the **code repair (bug fix)** task using the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) dataset.

## 📁 Directory Structure

```
.
├── BugFix/      # Original CodeXGLUE BugFix dataset (split into train/test/eval)
│   ├── eval/
│   ├── test/
│   └── train/
│
├── Dataset/sft_data_modified/      # Processed dataset for supervised fine-tuning
│   ├── eval.jsonl                  # Eval set (JSONL format)
│   └── train.jsonl                 # Train set (JSONL format)
│
├── Utils/
│   ├── sft_train.py               # Script to fine-tune the model
│   └──create_sft_dataset.sh      # Wrapper to generate eval/train JSONL files
├── create_sft_dataset.py      # Script to convert raw dataset into SFT-ready format
├── inference_CR.py            # Script to run inference on test set
├── run_inference_CR.sh        # Shell script to launch inference
└── sft_train.sh               # Shell script to launch fine-tuning
```

## 🧪 Experiment Details

### ✅ Dataset

- **Source**: [CodeXGLUE BugFix Dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-code-trans/BugFix)
- **Objective**: Fix buggy C++ code snippets given the correct version.
- **Preprocessing**: 
  - Converted dataset to supervised format using `Utils/create_sft_dataset.sh`.
  - Each datapoint is a pair of ⟨buggy, fixed⟩ code stored in JSONL format.

### 📊 Dataset Statistics 
**Dataset Size (Small Examples &lt; 50 Tokens):** 
- **Training Set:** 46,680 examples 
- **Validation Set:** 5,835 examples 
- **Test Set:** 5,835 examples

## 🧠 Supervised Fine-Tuning

### 🔧 Training

Use the script `Utils/sft_train.sh`:

```bash
CUDA_VISIBLE_DEVICES="3" python3 Utils/sft_train.py \
  --model_size "Qwen32b" \
  --output_dir "/path/to/save/model" \
  --train_dataset_path "Dataset/sft_data_modified/train.jsonl" \
  --eval_dataset_path "Dataset/sft_data_modified/eval.jsonl"
```

### 🔍 Inference

Use the script `Utils/run_inference_CR.sh`:

```bash
python3 inference_CR.py \
  --model_size "Qwen32b" \
  --adapter_path "Model/CodeRepairModifiedPrompt/final_checkpoint" \
  --device "cuda:4" \
  --output_file_path "BugFix/FineTuneResults/results.json" \
  --max_length "4096" \
  --buggy_path "BugFix/test/buggy.txt" \
  --fixed_path "BugFix/test/fixed.txt"
```

## 📈 Goal

Evaluate the fine-tuned model on unseen buggy code samples to determine if the generated output fixes the bug — **test cases serve as the final correctness measure**.

## 📈 Task Summary & Results 
### 🧠 Model 
- **Model Name:** `Qwen/QwQ-32B` 
  
### 🧪 Code Repair Task Summary – CodeXGLUE Dataset 
**Before Supervised Fine-Tuning:** 
- BLEU Score: `0.5258` 
- CodeBLEU Score: `0.4630` 

**After Supervised Fine-Tuning:** 
- BLEU Score: `0.5845` 
- CodeBLEU Score: `0.5913`
