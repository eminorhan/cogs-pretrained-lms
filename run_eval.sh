#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=t5_cogs_eval
#SBATCH --output=t5_cogs_eval_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_seq2seq.py \
    --model_name_or_path tmp/checkpoint-150000 \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --validation_file data/gen.json \
    --test_file data/gen.json \
    --output_dir tmp_eval \
    --per_device_eval_batch_size=1 \
    --do_predict \
    --predict_with_generate \
    --max_target_length 2000

echo "Done"
