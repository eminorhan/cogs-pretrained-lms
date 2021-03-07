#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --qos=batch
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=t5_cogs
#SBATCH --output=t5_cogs_%A_%a.out
#SBATCH --error=t5_cogs_%A_%a.err
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=lr2715@courant.nyu.edu
#SBATCH --nodelist=weaver5

module load python-3.7
module purge
module load cuda-10.1
source pretrain/bin/activate

srun python3.7 -u /misc/vlgscratch4/LakeGroup/Laura/cogs-pretrained-lms/run_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --train_file data/train.json \
    --validation_file data/gen.json \
    --output_dir tmp \
    --per_device_train_batch_size=100 \
    --per_device_eval_batch_size=10 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 25000 \
    --max_target_length 2000 \
    --num_train_epochs 83 \
    --max_val_samples 500 \
    --learning_rate 0.001 \
    --evaluation_strategy steps \
    --eval_steps 5000 &> test_run.txt

echo "Done"
