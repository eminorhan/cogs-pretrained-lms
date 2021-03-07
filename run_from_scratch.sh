#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=t5_cogs
#SBATCH --output=t5_cogs_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_seq2seq.py \
    --model_name_or_path t5-small \
    --use_pretrained_weights False \
    --do_train \
    --do_eval \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --train_file data/train.json \
    --validation_file data/gen.json \
    --output_dir tmp_t5_small_scratch \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 25000 \
    --max_target_length 2000 \
    --num_train_epochs 10 \
    --max_val_samples 128

echo "Done"
