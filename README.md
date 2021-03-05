# Semantic interpretation with pretrained language models

First, install huggingface [transformers](https://huggingface.co/transformers/installation.html#installing-from-source) and [datasets](https://huggingface.co/docs/datasets/installation.html#installing-from-source) libraries from source. The fine-tuning code here is copied from the [huggingface examples](https://github.com/huggingface/transformers/blob/master/examples/seq2seq) repository.

I've preprocessed and saved all the [COGS dataset](https://github.com/najoungkim/COGS) splits in `json` files inside the [data](https://github.com/eminorhan/cogs-pretrained-lms/tree/master/data) directory. To fine-tune a small pretrained T5 model on the COGS training set and then evaluate on the generalization set, run e.g.: 

```
python run_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --train_file data/train.json \
    --validation_file data/gen.json \
    --output_dir tmp \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 10000 \
    --num_train_epochs 10
```

For evaluating a saved checkpoint on a dataset, run e.g.:

```
python run_seq2seq.py \
    --model_name_or_path *PATH_TO_SAVED_CHKPT* \  
    --do_predict \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --validation_file data/gen.json \
    --test_file data/gen.json \
    --output_dir *OUTPUT_PATH* \  # results will be saved here
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --max_target_length 2000
```
