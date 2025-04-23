TRAIN_PATH="/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/train"
EVAL_PATH="/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/eval"
TEST_PATH="/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/test"
OUTPUT_PATH="/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Dataset/sft_data_modified"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

python3 Utils/create_sft_dataset.py \
    --train_path "$TRAIN_PATH" \
    --eval_path "$EVAL_PATH" \
    --test_path "$TEST_PATH" \
    --output_path "$OUTPUT_PATH"