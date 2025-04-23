#!/bin/bash

python3 /raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/inference_CR.py \
  --model_size "Qwen32b" \
  --adapter_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Model/CodeRepairModifiedPrompt/final_checkpoint" \
  --device "cuda:4" \
  --output_file_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/FineTuneResults/results.json" \
  --max_length "4096" \
  --buggy_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/test/buggy.txt" \
  --fixed_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/test/fixed.txt"
