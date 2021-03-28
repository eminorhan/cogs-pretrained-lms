#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0

module purge
module load cuda-10.1
module load python3.7
source pretrain/bin/activate

srun python3.7 -u /misc/vlgscratch4/LakeGroup/Laura/cogs-pretrained-lms/run_seq2seq.py \
    --model_name_or_path ${1} \
    --use_pretrained_weights ${2} \
    --do_train \
    --do_eval \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --train_file data/train.json \
    --validation_file data/dev.json \
    --output_dir ${3} \
    --per_device_train_batch_size ${4} \
    --per_device_eval_batch_size ${4} \
    --overwrite_output_dir \
    --save_steps 250000 \
    --max_target_length 2000 \
    --learning_rate 0.001 \
    --num_train_epochs ${5} \
    --evaluation_strategy steps \
    --eval_steps ${6} \
    --target_lang fr &>> ${3}/${7}_train.txt

echo "Done"
