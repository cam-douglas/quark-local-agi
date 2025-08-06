#!/usr/bin/env python3
"""
evaluate.py

Evaluates a fine-tuned model on its task-specific validation set.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import metrics as metrics_module
import datasets as ds_module
from train import tokenize_intent, tokenize_ner, tokenize_summarization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['intent', 'ner', 'summarization'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_name')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    # Load tokenizer and model
    if args.task == 'intent':
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
        dataset = ds_module.load_intent_dataset(args.dataset_name, args.subset)
        tokenized = dataset.map(lambda ex: tokenize_intent(ex, tokenizer), batched=True)
        tokenized.set_format('torch')
        metric_fn = metrics_module.compute_metrics_intent
        eval_dataset = tokenized['validation']
    elif args.task == 'ner':
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
        dataset = ds_module.load_ner_dataset(args.dataset_name)
        tokenized = dataset.map(lambda ex: tokenize_ner(ex, tokenizer), batched=True)
        tokenized.set_format('torch')
        metric_fn = metrics_module.compute_metrics_ner
        eval_dataset = tokenized['validation']
    elif args.task == 'summarization':
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
        dataset = ds_module.load_summarization_dataset(args.dataset_name, args.subset)
        tokenized = dataset.map(lambda ex: tokenize_summarization(ex, tokenizer), batched=True)
        tokenized.set_format('torch')
        metric_fn = lambda eval_pred: metrics_module.compute_metrics_summarization(eval_pred, tokenizer)
        eval_dataset = tokenized['validation']

    training_args = TrainingArguments(
        output_dir='./eval',
        per_device_eval_batch_size=args.batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metric_fn
    )

    metrics = trainer.evaluate()
    print(metrics)

if __name__ == '__main__':
    main()

