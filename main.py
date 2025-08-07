#!/usr/bin/env python3
"""
Quark AI System Main Entry Point
Starts the Quark AI System with health check endpoints
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quark_main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def start_health_server():
    """Start the health check server in a separate thread"""
    try:
        from web.health_check import app
        logger.info("Starting health check server on port 8000...")
        app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")

def start_quark_system():
    """Start the main Quark AI System"""
    try:
        logger.info("Starting Quark AI System...")
        
        # Import and initialize Quark components
        from core.orchestrator import Orchestrator
        from agents.negotiation_agent import NegotiationAgent
        from agents.explainability_agent import ExplainabilityAgent
        
        # Initialize agents
        negotiation_agent = NegotiationAgent()
        explainability_agent = ExplainabilityAgent()
        
        # Initialize orchestrator
        orchestrator = Orchestrator()
        
        logger.info("Quark AI System initialized successfully")
        
        # Keep the system running
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Failed to start Quark system: {e}")
        raise

def main():
    """Main entry point"""
    logger.info("🚀 Starting Quark AI System...")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Start health server in a separate thread
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Wait a moment for health server to start
    time.sleep(2)
    
    # Start main Quark system
    try:
        start_quark_system()
    except KeyboardInterrupt:
        logger.info("Shutting down Quark AI System...")
    except Exception as e:
        logger.error(f"Quark AI System error: {e}")
        raise

if __name__ == "__main__":
    main() 