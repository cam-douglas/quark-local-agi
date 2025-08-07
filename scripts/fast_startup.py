#!/usr/bin/env python3
"""
Fast Quark Startup Script
Achieves millisecond startup times using preloaded models and parallel initialization
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
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastQuarkStartup:
    """Ultra-fast Quark startup with millisecond response times"""
    
    def __init__(self):
        self.startup_time = None
        self.ready_time = None
        self.components = {}
        self.cache_dir = Path("models/cache")
        self.ready_file = Path("quark.ready")
        
    def load_cached_models(self):
        """Load models from cache for instant access"""
        logger.info("Loading cached models...")
        
        # Load essential models from cache
        essential_models = ['gpt2', 'bert-base-uncased', 'sentence-transformers/all-MiniLM-L6-v2']
        
        for model_name in essential_models:
            cache_path = self.cache_dir / f"{model_name}_cache.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        model_data = pickle.load(f)
                    self.components[f"model_{model_name}"] = model_data
                    logger.info(f"Loaded cached model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load cached model {model_name}: {e}")
        
        logger.info(f"Loaded {len([k for k in self.components.keys() if k.startswith('model_')])} cached models")
    
    def initialize_agents_parallel(self):
        """Initialize all agents in parallel"""
        logger.info("Initializing agents in parallel...")
        
        agent_configs = [
            ('negotiation_agent', 'agents.negotiation_agent', 'NegotiationAgent'),
            ('explainability_agent', 'agents.explainability_agent', 'ExplainabilityAgent'),
            ('orchestrator', 'core.orchestrator', 'Orchestrator'),
            ('health_server', 'web.health_check', 'app')
        ]
        
        def init_agent(config):
            name, module_path, class_name = config
            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                
                if name == 'health_server':
                    # Start health server in background
                    def run_server():
                        agent_class.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
                    
                    thread = threading.Thread(target=run_server, daemon=True)
                    thread.start()
                    return name, thread
                else:
                    agent = agent_class()
                    return name, agent
                    
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
                return name, None
        
        # Initialize all agents in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(init_agent, config) for config in agent_configs]
            
            for future in as_completed(futures):
                name, component = future.result()
                if component is not None:
                    self.components[name] = component
                    logger.info(f"Initialized: {name}")
        
        logger.info(f"Initialized {len([k for k in self.components.keys() if not k.startswith('model_')])} components")
    
    def start_health_monitor(self):
        """Start health monitoring"""
        def health_monitor():
            while True:
                try:
                    # Check if health endpoint is responding
                    import requests
                    response = requests.get('http://localhost:8000/health', timeout=1)
                    if response.status_code == 200:
                        logger.info("Health check: OK")
                    else:
                        logger.warning("Health check: Failed")
                except Exception as e:
                    logger.warning(f"Health check error: {e}")
                
                time.sleep(30)  # Check every 30 seconds
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        self.components['health_monitor'] = health_thread
    
    def create_ready_signal(self):
        """Create ready signal file"""
        self.ready_file.touch()
        logger.info("Created ready signal")
    
    def start(self) -> Dict[str, Any]:
        """Start Quark with ultra-fast startup"""
        self.startup_time = time.time()
        logger.info("🚀 Starting Quark with ultra-fast startup...")
        
        # Phase 1: Load cached models (instant)
        self.load_cached_models()
        
        # Phase 2: Initialize components in parallel
        self.initialize_agents_parallel()
        
        # Phase 3: Start health monitoring
        self.start_health_monitor()
        
        # Phase 4: Create ready signal
        self.create_ready_signal()
        
        self.ready_time = time.time()
        startup_duration = (self.ready_time - self.startup_time) * 1000  # Convert to milliseconds
        
        logger.info(f"✅ Quark started in {startup_duration:.2f}ms")
        
        return {
            'startup_time': self.startup_time,
            'ready_time': self.ready_time,
            'duration_ms': startup_duration,
            'components': list(self.components.keys()),
            'status': 'ready'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'running': self.ready_file.exists(),
            'components': list(self.components.keys()),
            'startup_duration_ms': (self.ready_time - self.startup_time) * 1000 if self.ready_time else None
        }

def main():
    """Main entry point for fast startup"""
    startup = FastQuarkStartup()
    result = startup.start()
    
    print(f"✅ Quark started in {result['duration_ms']:.2f}ms")
    print(f"📦 Components: {', '.join(result['components'])}")
    print(f"🌐 Health check: http://localhost:8000/health")
    print(f"📊 Ready check: http://localhost:8000/ready")
    print(f"⚡ Ultra-fast startup achieved!")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Quark...")

if __name__ == "__main__":
    main() 