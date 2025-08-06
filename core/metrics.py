# File: metrics.py
#!/usr/bin/env python3
"""
metrics.py

Basic metrics & logging for the Meta-Model AI Assistant.
- Counts tokens in/out
- Logs timing, token usage, and category to a rolling JSON-lines file
"""
import time
import json
import os
from transformers import AutoTokenizer

# Ensure a logs/ folder exists next to this file
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Path to our JSON-lines log
LOG_PATH = os.path.join(LOG_DIR, "assistant.log")

# Touch the log file so it always exists on disk
if not os.path.exists(LOG_PATH):
    open(LOG_PATH, "w").close()

# Use a lightweight tokenizer for counting
_TOKENIZER = AutoTokenizer.from_pretrained("t5-small")


def count_tokens(text: str) -> int:
    """Return the number of input tokens (no special tokens)."""
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def log_metric(entry: dict):
    """Append a JSON-line entry to the log file."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

