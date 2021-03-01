#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=t5_cogs
#SBATCH --output=t5_cogs_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --train_file /data/train.json \
    --validation_file /data/gen.json \
    --output_dir /tmp \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 24000 \
    --max_val_samples 21000

echo "Done"