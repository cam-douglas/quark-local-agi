"""
Health Check Endpoint for Quark AI System
Simple HTTP endpoint to verify Quark is running and ready
"""

from flask import Flask, jsonify
import os
import sys
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Quark AI System',
        'version': '1.0.0'
    })

@app.route('/ready')
def ready_check():
    """Readiness check endpoint"""
    return jsonify({
        'status': 'ready',
        'timestamp': datetime.now().isoformat(),
        'service': 'Quark AI System',
        'ready': True
    })

@app.route('/')
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'Quark AI System is running',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': '/health',
            'ready': '/ready'
        }
    })

if __name__ == '__main__':
    print("Starting Quark Health Check Server...")
    print("Health check available at: http://localhost:8000/health")
    print("Ready check available at: http://localhost:8000/ready")
    app.run(host='0.0.0.0', port=8000, debug=False) 