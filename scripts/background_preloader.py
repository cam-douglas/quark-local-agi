#!/usr/bin/env python3
"""
Background Model Preloader for Quark AI System
Continuously monitors and updates models without blocking startup
"""

import os
import sys
import time
import asyncio
import threading
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.streaming_model_preloader import StreamingModelPreloader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackgroundModelPreloader:
    """Background model preloader for continuous updates"""
    
    def __init__(self):
        self.preloader = StreamingModelPreloader()
        self.running = False
        self.update_interval = 3600  # 1 hour
        
    async def continuous_preload(self):
        """Continuously preload models in background"""
        logger.info("🔄 Starting background model preloader...")
        
        while self.running:
            try:
                # Check for model updates
                logger.info("🔍 Checking for model updates...")
                
                # Run preload in background
                successful, total = await self.preloader.preload_essential_models()
                
                if successful > 0:
                    logger.info(f"✅ Background preload: {successful}/{total} models updated")
                else:
                    logger.info("✅ All models up to date")
                
                # Wait before next check
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Background preload error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def start(self):
        """Start background preloader"""
        self.running = True
        
        def run_async():
            asyncio.run(self.continuous_preload())
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        logger.info("✅ Background preloader started")
        return thread
    
    def stop(self):
        """Stop background preloader"""
        self.running = False
        logger.info("🛑 Background preloader stopped")

def main():
    """Main entry point for background preloader"""
    preloader = BackgroundModelPreloader()
    
    try:
        # Start background preloader
        thread = preloader.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
        preloader.stop()
    except Exception as e:
        logger.error(f"Background preloader error: {e}")

if __name__ == "__main__":
    main() 