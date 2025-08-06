#!/usr/bin/env bash
# Convenience script to run training
python3 train.py --task intent --model_name distilbert-base-uncased-finetuned-mnli --dataset_name glue --subset mnli

