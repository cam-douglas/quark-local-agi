#!/usr/bin/env python3
"""
Preload all models so they're cached before shell startup.
Run this once after you install your env.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Preload all models used in the project."""
    print("🔥 Preloading models for faster startup...")
    
    try:
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
        import spacy
        
        print("📥 Loading zero-shot classification model...")
        pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        print("📥 Loading intent classification model...")
        pipeline("text-classification", model="typeform/distilbert-base-uncased-mnli")
        
        print("📥 Loading named entity recognition model...")
        pipeline("token-classification", model="elastic/distilbert-base-cased-finetuned-conll03-english")
        
        print("📥 Loading sentiment analysis model...")
        pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        print("📥 Loading semantic embedding model...")
        SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        
        print("📥 Loading fast embedding model...")
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("📥 Loading paraphrase detection model...")
        SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")
        
        print("📥 Loading text generation model...")
        pipeline("text2text-generation", model="google/flan-t5-small")
        
        print("📥 Loading text generation model (alternative)...")
        pipeline("text2text-generation", model="t5-small")
        
        print("📥 Loading creative writing model...")
        pipeline("text-generation", model="distilgpt2")
        
        print("📥 Loading summarization model...")
        pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        print("📥 Loading translation model...")
        pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
        
        print("📥 Loading SpaCy English model...")
        spacy.load("en_core_web_sm")
        
        print("✅ All models preloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error preloading models: {e}")
        print("💡 This is normal for first-time setup. Models will be downloaded when needed.")

if __name__ == "__main__":
    main()

