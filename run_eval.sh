#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=t5_cogs
#SBATCH --output=t5_cogs_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_seq2seq.py \
    --model_name_or_path tmp/checkpoint-150000 \  # path to saved model and configs 
    --do_predict \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --validation_file data/gen.json \
    --test_file data/gen.json \
    --output_dir tmp2 \  # results will be saved here
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --max_target_length 2000 \
    --max_test_samples 8

echo "Done"
