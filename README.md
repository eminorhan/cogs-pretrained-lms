# Semantic interpretation with pretrained language models

First, install huggingface [transformers](https://huggingface.co/transformers/installation.html#installing-from-source) and [datasets](https://huggingface.co/docs/datasets/installation.html#installing-from-source) libraries from source. The code here is adapted from the [huggingface examples](https://github.com/huggingface/transformers/blob/master/examples/seq2seq) repository.

I've preprocessed and saved all the [COGS dataset](https://github.com/najoungkim/COGS) splits in `json` files inside the [data](https://github.com/eminorhan/cogs-pretrained-lms/tree/master/data) directory. To fine-tune a small pretrained T5 model on the COGS training set and then evaluate on the generalization set, run e.g.: 

```
python run_seq2seq.py \
    --model_name_or_path t5-small \
    --use_pretrained_weights True \
    --do_train \
    --do_predict \
    --task translation_en_to_mentalese \
    --source_prefix "translate English to Mentalese: " \
    --train_file data/train.json \
    --test_file data/gen.json \
    --output_dir tmp_t5_small \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --save_steps 250000 \
    --max_target_length 2000 \
    --num_train_epochs 100 \
    --predict_with_generate
```

## Results
Detailed results are in the [`results`](https://github.com/eminorhan/cogs-pretrained-lms/tree/master/results) folder.

| model | generalization accuracy | training loss | epochs | batch size | 
| ----- |:-----------------------:|:-------------:|:------:|:----------:|
| `t5_small_scratch` | 0.48       | 0.0001        | 100    | 72         |
| `t5_base_scratch`  | 0.27       | 0.0003        | 50     | 72         |
| `t5_large_scratch` | ---        | ---           | ---    | ---        |
