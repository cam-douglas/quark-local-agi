#!/usr/bin/env python3
"""
train.py

Fine-tunes specified model on a selected task: intent, ner, or summarization.
"""
import argparse
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments
)
from datasets import ClassLabel, Sequence
import datasets
import numpy as np
import torch

import datasets as ds_module
import metrics as metrics_module


def tokenize_intent(examples, tokenizer):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True)


def tokenize_ner(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def tokenize_summarization(examples, tokenizer):
    inputs = tokenizer(examples['article'], truncation=True, padding='max_length', max_length=512)
    outputs = tokenizer(examples['highlights'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['intent', 'ner', 'summarization'], required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset_name')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    # Load tokenizer and model
    if args.task == 'intent':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
        dataset = ds_module.load_intent_dataset(args.dataset_name, args.subset)
        tokenized = dataset.map(lambda ex: tokenize_intent(ex, tokenizer), batched=True)
        tokenized = tokenized.remove_columns(['prompt', 'label'])
        tokenized.set_format('torch')
        metric_fn = metrics_module.compute_metrics_intent
    elif args.task == 'ner':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=dataset.features['ner_tags'].feature.num_classes)
        dataset = ds_module.load_ner_dataset(args.dataset_name)
        tokenized = dataset.map(lambda ex: tokenize_ner(ex, tokenizer), batched=True)
        tokenized.set_format('torch')
        metric_fn = metrics_module.compute_metrics_ner
    elif args.task == 'summarization':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        dataset = ds_module.load_summarization_dataset(args.dataset_name, args.subset)
        tokenized = dataset.map(lambda ex: tokenize_summarization(ex, tokenizer), batched=True)
        tokenized.set_format('torch')
        metric_fn = lambda eval_pred: metrics_module.compute_metrics_summarization(eval_pred, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy' if args.task=='intent' else 'rougeL'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        compute_metrics=metric_fn
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()

