#!/usr/bin/env python3
"""
download_models.py

Downloads specified models from Hugging Face and GitHub into the models/ directory.
Supports concurrent downloads and a failsafe fallback mechanism for Hugging Face models.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure huggingface_hub is available
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Missing dependency 'huggingface_hub'. Please activate your venv and run:\n  python3 -m pip install huggingface_hub")
    sys.exit(1)

# --------------------------------------
# CONFIGURATION: All models used in the project
# --------------------------------------
HF_MODELS = [
    # Intent classification and NLU
    "facebook/bart-large-mnli",                                # Zero-shot classification
    "typeform/distilbert-base-uncased-mnli",                  # Intent classification
    "elastic/distilbert-base-cased-finetuned-conll03-english",# Named entity recognition
    "dslim/bert-base-NER",                                    # Named entity recognition (alternative)
    "distilbert-base-uncased-finetuned-sst-2-english",        # Sentiment analysis
    
    # Semantic embeddings and retrieval
    "sentence-transformers/all-distilroberta-v1",             # Semantic embeddings for retrieval
    "sentence-transformers/all-MiniLM-L6-v2",                 # Fast embeddings
    "sentence-transformers/paraphrase-albert-small-v2",       # Paraphrase detection
    
    # Text generation and reasoning
    "google/flan-t5-small",                                   # Reasoning, summarization, translation
    "t5-small",                                               # Text-to-text generation
    "distilgpt2",                                             # Creative writing
    "sshleifer/distilbart-cnn-12-6",                         # Summarization
    
    # Translation
    "Helsinki-NLP/opus-mt-en-de",                            # English to German translation
]

HF_MODEL_ALTERNATIVES = {
    "google/flan-t5-small": ["t5-small"],
    "facebook/bart-large-mnli": ["typeform/distilbert-base-uncased-mnli"],
    "elastic/distilbert-base-cased-finetuned-conll03-english": ["dslim/bert-base-NER"],
}

# GitHub model repositories (org/repo or URL)
GITHUB_REPOS = [
    "whoosh-community/whoosh",    # Keyword-based document search
    "deepset-ai/haystack",        # RAG & FAQ lookup pipelines
    "chroma-core/chroma",         # Vector DB for long-term memory
]

# Directory to store all downloaded models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


def ensure_models_dir():
    """Create the models directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"📁 Models directory: {MODELS_DIR}")


def download_single_model(primary_id: str):
    """
    Attempt to download a primary HF model; if it fails, try alternatives.
    Returns the id of the successfully downloaded model or None if all fail.
    """
    candidates = [primary_id] + HF_MODEL_ALTERNATIVES.get(primary_id, [])
    for model_id in candidates:
        try:
            print(f"📥 Downloading Hugging Face model '{model_id}'...")
            snapshot_download(
                repo_id=model_id,
                cache_dir=MODELS_DIR,
                local_dir=os.path.join(MODELS_DIR, model_id.replace('/', '_')),
                resume_download=True,
            )
            print(f"✅ Successfully downloaded '{model_id}'")
            return model_id
        except Exception as e:
            print(f"❌ Failed to download '{model_id}': {e}")
    print(f"⚠️  Skipping model '{primary_id}' after all attempts failed.")
    return None


def download_hf_models(concurrency: int = None):
    """Download each HF model concurrently with fallbacks."""
    concurrency = concurrency or min(4, len(HF_MODELS))
    print(f"🚀 Starting download of {len(HF_MODELS)} models with {concurrency} workers...")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(download_single_model, m): m for m in HF_MODELS}
        successful_downloads = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                successful_downloads += 1
    
    print(f"✅ Downloaded {successful_downloads}/{len(HF_MODELS)} models successfully")


def clone_single_repo(repo: str):
    """Clone or update a single GitHub repository."""
    url = repo if repo.startswith("http") else f"https://github.com/{repo}.git"
    repo_name = os.path.splitext(os.path.basename(url))[0]
    dest_path = os.path.join(MODELS_DIR, repo_name)
    try:
        if os.path.isdir(dest_path):
            print(f"🔄 Updating existing repo '{repo_name}'...")
            subprocess.run(["git", "-C", dest_path, "pull"], check=True, capture_output=True)
        else:
            print(f"📥 Cloning repository '{url}' into '{dest_path}'...")
            subprocess.run(["git", "clone", url, dest_path], check=True, capture_output=True)
        print(f"✅ Repository '{repo_name}' ready.")
        return True
    except Exception as e:
        print(f"❌ Failed to clone '{repo}': {e}")
        return False


def clone_github_repos(concurrency: int = None):
    """Clone each GitHub repository concurrently."""
    concurrency = concurrency or min(2, len(GITHUB_REPOS))
    print(f"🚀 Starting clone of {len(GITHUB_REPOS)} repositories...")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(clone_single_repo, repo): repo for repo in GITHUB_REPOS}
        successful_clones = 0
        for future in as_completed(futures):
            if future.result():
                successful_clones += 1
    
    print(f"✅ Cloned {successful_clones}/{len(GITHUB_REPOS)} repositories successfully")


def main():
    """Main download function."""
    print("🚀 Quark AI Assistant - Model Downloader")
    print("=" * 50)
    
    ensure_models_dir()
    
    # Download Hugging Face models
    print("\n📥 Downloading Hugging Face models...")
    download_hf_models()
    
    # Clone GitHub repositories
    print("\n📥 Cloning GitHub repositories...")
    clone_github_repos()
    
    print("\n✅ Model download complete!")
    print("🎯 You can now run the AI assistant with: python scripts/ai_assistant.sh")


if __name__ == "__main__":
    main()

