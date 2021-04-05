#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=t5_cogs
#SBATCH --output=t5_cogs_%A_%a.out

module purge
module load cuda-10.1

((MAX_TRAIN_DEPTH=10))
((MIN_TEST_DEPTH=MAX_TRAIN_DEPTH+1))

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_translation.py \
    --model_name_or_path tmp_t5_small_pre_recursion_10 \
    --use_pretrained_weights True \
    --do_predict \
    --source_lang en \
    --target_lang en \
    --finetune_target_lang mentalese \
    --source_prefix "translate English to English: " \
    --test_file data/recursion_splits/test_recursion_depth_from_${MIN_TEST_DEPTH}_to_12.json \
     --validation_file data/recursion_splits/test_recursion_depth_from_${MIN_TEST_DEPTH}_to_12.json \
    --gen_conditions_file data/recursion_splits/conditions_recursion_depth_from_${MIN_TEST_DEPTH}_to_12.json \
    --output_dir tmp_t5_small_pre_recursion_${MAX_TRAIN_DEPTH}_eval \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --save_steps 250000 \
    --max_target_length 1024 \
    --predict_with_generate

echo "Done"
