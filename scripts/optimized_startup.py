#!/usr/bin/env python3
"""
Optimized Quark Startup Script
Implements model streaming, preloading, and intelligent caching for millisecond startup times
"""

import os
import sys
import time
import threading
import asyncio
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
import queue

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    """Intelligent model caching system"""
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
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
    
    def get_cache_key(self, model_name: str, model_config: Dict[str, Any]) -> str:
        """Generate cache key for model"""
        config_hash = hashlib.md5(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
        return f"{model_name}_{config_hash}"
    
    def is_cached(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """Check if model is cached"""
        cache_key = self.get_cache_key(model_name, model_config)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            # Check if cache is still valid (not older than 24 hours)
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                return True
        
        return False
    
    def save_to_cache(self, model_name: str, model_config: Dict[str, Any], model_data: Any):
        """Save model to cache"""
        cache_key = self.get_cache_key(model_name, model_config)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.cache_index[cache_key] = {
                'model_name': model_name,
                'config': model_config,
                'cached_at': datetime.now().isoformat(),
                'size': cache_path.stat().st_size
            }
            self._save_cache_index()
            logger.info(f"Cached model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to cache model {model_name}: {e}")
    
    def load_from_cache(self, model_name: str, model_config: Dict[str, Any]) -> Optional[Any]:
        """Load model from cache"""
        cache_key = self.get_cache_key(model_name, model_config)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    model_data = pickle.load(f)
                logger.info(f"Loaded model from cache: {model_name}")
                return model_data
            except Exception as e:
                logger.error(f"Failed to load cached model {model_name}: {e}")
        
        return None

class ModelStreamer:
    """Model streaming from cloud sources"""
    
    def __init__(self):
        self.streaming_queue = queue.Queue()
        self.streaming_thread = None
        self.cloud_sources = {
            'huggingface': 'https://huggingface.co/api/models',
            'openai': 'https://api.openai.com/v1/models',
            'anthropic': 'https://api.anthropic.com/v1/models'
        }
    
    def start_streaming(self):
        """Start model streaming thread"""
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()
        logger.info("Model streaming started")
    
    def _streaming_worker(self):
        """Background worker for model streaming"""
        while True:
            try:
                model_request = self.streaming_queue.get(timeout=1)
                self._stream_model(model_request)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
    
    def _stream_model(self, model_request: Dict[str, Any]):
        """Stream a model from cloud"""
        model_name = model_request['name']
        source = model_request.get('source', 'huggingface')
        
        logger.info(f"Streaming model: {model_name} from {source}")
        
        # Simulate streaming download
        # In production, this would use actual streaming APIs
        time.sleep(0.1)  # Simulate network delay
        
        logger.info(f"Model streamed: {model_name}")
    
    def request_model(self, model_name: str, source: str = 'huggingface'):
        """Request a model to be streamed"""
        self.streaming_queue.put({
            'name': model_name,
            'source': source,
            'timestamp': datetime.now().isoformat()
        })

class OptimizedQuarkStartup:
    """Optimized Quark startup with millisecond response times"""
    
    def __init__(self):
        self.model_cache = ModelCache()
        self.model_streamer = ModelStreamer()
        self.startup_time = None
        self.ready_time = None
        self.initialized_components = {}
        
    def preload_essential_models(self):
        """Preload essential models in background"""
        essential_models = [
            {'name': 'text-generation', 'config': {'model_type': 'gpt2'}},
            {'name': 'text-classification', 'config': {'model_type': 'bert'}},
            {'name': 'embeddings', 'config': {'model_type': 'sentence-transformers'}}
        ]
        
        logger.info("Preloading essential models...")
        
        # Start streaming in background
        self.model_streamer.start_streaming()
        
        # Preload models in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for model in essential_models:
                if not self.model_cache.is_cached(model['name'], model['config']):
                    # Request streaming
                    self.model_streamer.request_model(model['name'])
                
                # Preload from cache if available
                future = executor.submit(self._preload_model, model)
                futures.append(future)
            
            # Wait for all preloads to complete
            for future in futures:
                future.result()
        
        logger.info("Essential models preloaded")
    
    def _preload_model(self, model: Dict[str, Any]):
        """Preload a single model"""
        model_name = model['name']
        model_config = model['config']
        
        # Try to load from cache first
        cached_model = self.model_cache.load_from_cache(model_name, model_config)
        if cached_model:
            self.initialized_components[model_name] = cached_model
            return
        
        # Simulate model loading (in production, this would load actual models)
        logger.info(f"Loading model: {model_name}")
        time.sleep(0.05)  # Simulate loading time
        
        # Create mock model data
        mock_model = {
            'name': model_name,
            'config': model_config,
            'loaded_at': datetime.now().isoformat(),
            'status': 'ready'
        }
        
        # Cache the model
        self.model_cache.save_to_cache(model_name, model_config, mock_model)
        self.initialized_components[model_name] = mock_model
    
    def initialize_core_components(self):
        """Initialize core Quark components"""
        logger.info("Initializing core components...")
        
        # Initialize agents in parallel
        agent_futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            agent_futures.append(executor.submit(self._init_negotiation_agent))
            agent_futures.append(executor.submit(self._init_explainability_agent))
            agent_futures.append(executor.submit(self._init_orchestrator))
            agent_futures.append(executor.submit(self._init_health_server))
        
        # Wait for all agents to initialize
        for future in agent_futures:
            future.result()
        
        logger.info("Core components initialized")
    
    def _init_negotiation_agent(self):
        """Initialize negotiation agent"""
        try:
            from agents.negotiation_agent import NegotiationAgent
            agent = NegotiationAgent()
            self.initialized_components['negotiation_agent'] = agent
            logger.info("Negotiation agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize negotiation agent: {e}")
    
    def _init_explainability_agent(self):
        """Initialize explainability agent"""
        try:
            from agents.explainability_agent import ExplainabilityAgent
            agent = ExplainabilityAgent()
            self.initialized_components['explainability_agent'] = agent
            logger.info("Explainability agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize explainability agent: {e}")
    
    def _init_orchestrator(self):
        """Initialize orchestrator"""
        try:
            from core.orchestrator import Orchestrator
            orchestrator = Orchestrator()
            self.initialized_components['orchestrator'] = orchestrator
            logger.info("Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
    
    def _init_health_server(self):
        """Initialize health server"""
        try:
            from web.health_check import app
            # Start health server in background thread
            def run_health_server():
                app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
            
            health_thread = threading.Thread(target=run_health_server, daemon=True)
            health_thread.start()
            self.initialized_components['health_server'] = health_thread
            logger.info("Health server initialized")
        except Exception as e:
            logger.error(f"Failed to initialize health server: {e}")
    
    def start(self) -> Dict[str, Any]:
        """Start Quark with optimized startup"""
        self.startup_time = time.time()
        logger.info("🚀 Starting optimized Quark AI System...")
        
        # Phase 1: Preload essential models (parallel)
        self.preload_essential_models()
        
        # Phase 2: Initialize core components (parallel)
        self.initialize_core_components()
        
        # Phase 3: Final initialization
        self._final_initialization()
        
        self.ready_time = time.time()
        startup_duration = (self.ready_time - self.startup_time) * 1000  # Convert to milliseconds
        
        logger.info(f"✅ Quark started in {startup_duration:.2f}ms")
        
        return {
            'startup_time': self.startup_time,
            'ready_time': self.ready_time,
            'duration_ms': startup_duration,
            'components': list(self.initialized_components.keys()),
            'status': 'ready'
        }
    
    def _final_initialization(self):
        """Final initialization steps"""
        logger.info("Performing final initialization...")
        
        # Create ready file
        ready_file = Path("quark.ready")
        ready_file.touch()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Final initialization complete")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def background_worker():
            while True:
                try:
                    # Clean old cache entries
                    self._clean_cache()
                    # Update model streaming
                    self._update_streaming()
                    time.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logger.error(f"Background task error: {e}")
        
        background_thread = threading.Thread(target=background_worker, daemon=True)
        background_thread.start()
        logger.info("Background tasks started")
    
    def _clean_cache(self):
        """Clean old cache entries"""
        try:
            # Remove cache entries older than 7 days
            cache_dir = Path("models/cache")
            if cache_dir.exists():
                current_time = time.time()
                for cache_file in cache_dir.glob("*.pkl"):
                    if current_time - cache_file.stat().st_mtime > 604800:  # 7 days
                        cache_file.unlink()
                        logger.info(f"Removed old cache: {cache_file.name}")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def _update_streaming(self):
        """Update model streaming"""
        try:
            # Check for new model requests
            # In production, this would check for new model updates
            pass
        except Exception as e:
            logger.error(f"Streaming update error: {e}")

def main():
    """Main entry point for optimized startup"""
    startup = OptimizedQuarkStartup()
    result = startup.start()
    
    print(f"✅ Quark started in {result['duration_ms']:.2f}ms")
    print(f"📦 Components: {', '.join(result['components'])}")
    print(f"🌐 Health check: http://localhost:8000/health")
    print(f"📊 Ready check: http://localhost:8000/ready")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Quark...")

if __name__ == "__main__":
    main() 