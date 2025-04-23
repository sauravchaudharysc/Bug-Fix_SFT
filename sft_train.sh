CUDA_VISIBLE_DEVICES="3" python3 Utils/sft_train.py \
    --model_size "Qwen32b" \
    --output_dir "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Model//raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Model/CodeRepairModifiedPrompt" \
    --train_dataset_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Dataset/sft_data_modified/train.jsonl" \
    --eval_dataset_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Dataset/sft_data_modified/eval.jsonl"  