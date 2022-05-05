# Bridge the Commonsense Gap: A Self-Supervised Learning Approach for Story Cloze Test

Code and models for the paper under review.

## Requirements

- Python 3.6+
- PyTorch >= 0.4.1
- transformers = 2.1.1

## Getting started

Please run the model on two GPUs (1080Ti+) at least. We only give one combination form for each self-supervised task (Drop, Replace and TOV) in our code. The shell script give an example that trains vanilla BERT base model on ROCStories with TOV [[1],[2,3,4]], and then fine-tunes the resulting model on SCT-v1.0 test set and test on SCT-v1.5 dev set.

Download the pre-trained vanilla BERT and RoBERTa into the data file. For example:

'bert-base-uncased': 

Config: "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",

Model: "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",

Vocab: "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",

1. Train a vanilla BERT and a Multi-Choice Head on ROCStories

- We randomly split ROCStories into train set (80%) and dev set (20%). 
- File: /data/roc_train.csv and /data/roc_dev.csv
- Run shell script: sh run_train_on_rocstories.sh

2. Fine-tune the resulting model, which learn the commonsense from ROCStories, on the SCT task.

- File: /data/sct_1.0_test and /data/sct_1.5_val
- Run shell script: sh run_finetune_on_sct.sh







