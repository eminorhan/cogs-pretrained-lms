#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=m2m100_cogs
#SBATCH --output=m2m100_cogs_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_translation.py \
    --model_name_or_path facebook/m2m100_418M \
    --use_pretrained_weights True \
    --do_train \
    --do_predict \
    --source_lang en \
    --target_lang en \
    --finetune_target_lang mentalese \
    --train_file data/train.json \
    --test_file data/gen.json \
    --output_dir tmp_m2m100_418m_pre \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --save_steps 250000 \
    --max_target_length 2000 \
    --num_train_epochs 10 \
    --predict_with_generate

echo "Done"
