#!/usr/bin/env python3
"""
Datasets loader for fine-tuning tasks.
Provides functions to load intent, NER, and summarization datasets.
"""
from datasets import load_dataset


def load_intent_dataset(dataset_name: str = "glue", subset: str = "mnli"):
    """Load a classification dataset for intent detection."""
    return load_dataset(dataset_name, subset)


def load_ner_dataset(dataset_name: str = "conll2003"):
    """Load a token classification dataset for NER."""
    return load_dataset(dataset_name)


def load_summarization_dataset(dataset_name: str = "cnn_dailymail", subset: str = "3.0.0"):
    """Load a summarization dataset."""
    return load_dataset(dataset_name, subset)
