#!/usr/bin/env python3
"""
Streaming Model Preloader for Quark AI System
Downloads and caches models from cloud sources for millisecond startup
"""

import os
import sys
import time
import json
import hashlib
import requests
import threading
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingModelPreloader:
    """Streaming model preloader with intelligent caching"""
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "model_cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Cloud sources configuration
        self.cloud_sources = {
            'huggingface': {
                'base_url': 'https://huggingface.co/api/models',
                'download_url': 'https://huggingface.co/{model}/resolve/main/{file}',
                'api_token': os.getenv('HUGGINGFACE_TOKEN')
            },
            'openai': {
                'base_url': 'https://api.openai.com/v1/models',
                'api_token': os.getenv('OPENAI_API_KEY')
            }
        }
        
        # Essential models for Quark
        self.essential_models = [
            {
                'name': 'gpt2',
                'source': 'huggingface',
                'files': ['config.json', 'pytorch_model.bin', 'tokenizer.json'],
                'priority': 'high'
            },
            {
                'name': 'bert-base-uncased',
                'source': 'huggingface',
                'files': ['config.json', 'pytorch_model.bin', 'tokenizer.json'],
                'priority': 'high'
            },
            {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'source': 'huggingface',
                'files': ['config.json', 'pytorch_model.bin', 'tokenizer.json'],
                'priority': 'medium'
            }
        ]
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save cache index"""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def get_cache_key(self, model_name: str, file_name: str) -> str:
        """Generate cache key for model file"""
        return hashlib.md5(f"{model_name}_{file_name}".encode()).hexdigest()
    
    def is_cached(self, model_name: str, file_name: str) -> bool:
        """Check if model file is cached"""
        # First check if model exists in the main models directory
        model_dir = Path("models") / model_name.replace("/", "_")
        if model_dir.exists() and (model_dir / file_name).exists():
            logger.info(f"Found existing model: {model_name}/{file_name}")
            return True
        
        # Then check cache directory
        cache_key = self.get_cache_key(model_name, file_name)
        cache_path = self.cache_dir / f"{cache_key}.bin"
        
        if cache_path.exists():
            # Check if cache is still valid (not older than 7 days)
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < 604800:  # 7 days
                return True
        
        return False
    
    def check_existing_models(self) -> Dict[str, bool]:
        """Check which essential models already exist locally"""
        existing_models = {}
        
        for model in self.essential_models:
            model_name = model['name']
            model_dir = Path("models") / model_name.replace("/", "_")
            
            if model_dir.exists():
                # Check if all required files exist
                all_files_exist = all(
                    (model_dir / file_name).exists() 
                    for file_name in model['files']
                )
                existing_models[model_name] = all_files_exist
                if all_files_exist:
                    logger.info(f"✅ Model {model_name} already exists locally")
                else:
                    logger.info(f"⚠️  Model {model_name} exists but missing some files")
            else:
                existing_models[model_name] = False
                logger.info(f"❌ Model {model_name} not found locally")
        
        return existing_models
    
    async def stream_download(self, url: str, cache_path: Path, chunk_size: int = 8192):
        """Stream download a file"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async with aiofiles.open(cache_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
                    return True
                else:
                    logger.error(f"Failed to download {url}: {response.status}")
                    return False
    
    async def download_model_file(self, model_name: str, file_name: str, source: str = 'huggingface'):
        """Download a single model file"""
        cache_key = self.get_cache_key(model_name, file_name)
        cache_path = self.cache_dir / f"{cache_key}.bin"
        
        if self.is_cached(model_name, file_name):
            logger.info(f"Using cached file: {model_name}/{file_name}")
            return True
        
        logger.info(f"Downloading: {model_name}/{file_name}")
        
        if source == 'huggingface':
            url = f"https://huggingface.co/{model_name}/resolve/main/{file_name}"
            
            # Add authentication if token is available
            headers = {}
            if self.cloud_sources['huggingface']['api_token']:
                headers['Authorization'] = f"Bearer {self.cloud_sources['huggingface']['api_token']}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            async with aiofiles.open(cache_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            
                            # Update cache index
                            self.cache_index[cache_key] = {
                                'model_name': model_name,
                                'file_name': file_name,
                                'source': source,
                                'cached_at': time.time(),
                                'size': cache_path.stat().st_size
                            }
                            self._save_cache_index()
                            
                            logger.info(f"Downloaded: {model_name}/{file_name}")
                            return True
                        else:
                            logger.error(f"Failed to download {url}: {response.status}")
                            return False
            except Exception as e:
                logger.error(f"Download error for {model_name}/{file_name}: {e}")
                return False
        
        return False
    
    async def preload_essential_models(self):
        """Preload all essential models"""
        logger.info("🚀 Starting essential model preload...")
        
        # First check which models already exist locally
        existing_models = self.check_existing_models()
        
        # Create tasks only for missing model files
        tasks = []
        for model in self.essential_models:
            model_name = model['name']
            source = model['source']
            
            # Skip if model already exists locally
            if existing_models.get(model_name, False):
                logger.info(f"⏭️  Skipping {model_name} - already exists locally")
                continue
            
            for file_name in model['files']:
                # Only download if not cached
                if not self.is_cached(model_name, file_name):
                    task = self.download_model_file(model_name, file_name, source)
                    tasks.append(task)
                else:
                    logger.info(f"⏭️  Skipping {model_name}/{file_name} - already cached")
        
        if not tasks:
            logger.info("✅ All essential models already available locally")
            return len(self.essential_models), len(self.essential_models)
        
        # Execute all downloads concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful downloads
        successful = sum(1 for result in results if result is True)
        total = len(tasks)
        
        logger.info(f"✅ Preload complete: {successful}/{total} files downloaded")
        return successful, total
    
    def get_cached_model_path(self, model_name: str, file_name: str) -> Optional[Path]:
        """Get path to cached model file"""
        cache_key = self.get_cache_key(model_name, file_name)
        cache_path = self.cache_dir / f"{cache_key}.bin"
        
        if cache_path.exists():
            return cache_path
        return None
    
    def list_cached_models(self) -> Dict[str, Any]:
        """List all cached models"""
        cached_models = {}
        
        for cache_key, info in self.cache_index.items():
            model_name = info['model_name']
            if model_name not in cached_models:
                cached_models[model_name] = {
                    'files': [],
                    'total_size': 0,
                    'cached_at': info['cached_at']
                }
            
            cached_models[model_name]['files'].append(info['file_name'])
            cached_models[model_name]['total_size'] += info['size']
        
        return cached_models
    
    def cleanup_old_cache(self, max_age_days: int = 7):
        """Clean up old cache entries"""
        logger.info(f"🧹 Cleaning up cache older than {max_age_days} days...")
        
        current_time = time.time()
        max_age_seconds = max_age_days * 86400
        
        removed_count = 0
        for cache_key, info in list(self.cache_index.items()):
            if current_time - info['cached_at'] > max_age_seconds:
                cache_path = self.cache_dir / f"{cache_key}.bin"
                if cache_path.exists():
                    cache_path.unlink()
                    removed_count += 1
                
                del self.cache_index[cache_key]
        
        self._save_cache_index()
        logger.info(f"✅ Removed {removed_count} old cache entries")

async def main():
    """Main entry point"""
    preloader = StreamingModelPreloader()
    
    print("🚀 Quark Model Preloader")
    print("========================")
    
    # Show current cache status
    cached_models = preloader.list_cached_models()
    if cached_models:
        print(f"📦 Cached models: {len(cached_models)}")
        for model_name, info in cached_models.items():
            size_mb = info['total_size'] / (1024 * 1024)
            print(f"  • {model_name}: {len(info['files'])} files, {size_mb:.1f}MB")
    else:
        print("📦 No cached models found")
    
    print()
    
    # Preload essential models
    print("⏳ Preloading essential models...")
    start_time = time.time()
    
    successful, total = await preloader.preload_essential_models()
    
    duration = time.time() - start_time
    print(f"✅ Preload completed in {duration:.2f}s")
    print(f"📊 Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Cleanup old cache
    preloader.cleanup_old_cache()
    
    print()
    print("🎯 Ready for millisecond startup!")

if __name__ == "__main__":
    asyncio.run(main()) 