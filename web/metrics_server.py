#!/usr/bin/env python3
"""
Launch a HTTP server exposing /metrics for Prometheus scraping.
"""
from prometheus_client import start_http_server
import time

def main(port: int = 8001):
    """Start the Prometheus metrics endpoint."""
    start_http_server(port)
    print(f"📊 Metrics server listening on http://0.0.0.0:{port}/metrics")
    # Keep the process alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down metrics server.")

if __name__ == "__main__":
    main()

