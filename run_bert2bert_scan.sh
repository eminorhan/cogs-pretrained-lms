#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=bert2bert_cogs
#SBATCH --output=bert2bert_cogs_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/cogs-pretrained-lms/run_bert2bert.py \
    --encoder_model bert-base-cased \
    --decoder_model bert-base-cased \
    --do_train \
    --do_predict \
    --train_file data_scan/add_jump/train.json \
    --test_file data_scan/add_jump/test.json \
    --gen_conditions_file gen_conditions.txt \
    --output_dir tmp_bert2bert_scan_addjump_en \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --save_steps 250000 \
    --max_source_length 512 \
    --max_target_length 512 \
    --num_train_epochs 3 \
    --pad_to_max_length True \
    --fp16 \
    --predict_with_generate

echo "Done"
