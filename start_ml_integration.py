#!/usr/bin/env python3
"""
Startup script to run both the ML inference service and the backend server.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def run_ml_service():
    """Run the ML inference service."""
    print("🤖 Starting ML inference service...")
    
    # Change to ML model directory
    ml_dir = Path(__file__).parent / "ml-model"
    os.chdir(ml_dir)
    
    # Install dependencies if needed
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "inference/requirements.txt"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to install ML dependencies: {e}")
    
    # Start the ML API server
    try:
        subprocess.run([sys.executable, "inference/api_server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ML service failed: {e}")
    except KeyboardInterrupt:
        print("🛑 ML service stopped")

def run_backend():
    """Run the backend server."""
    print("🚀 Starting backend server...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Install dependencies if needed
    try:
        subprocess.run(["npm", "install"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to install backend dependencies: {e}")
    
    # Start the backend server
    try:
        subprocess.run(["npm", "run", "dev"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed: {e}")
    except KeyboardInterrupt:
        print("🛑 Backend stopped")

def main():
    """Main function to start both services."""
    print("🌟 Starting Lanternfly Tracker with ML Integration")
    print("=" * 50)
    
    # Start ML service in a separate thread
    ml_thread = threading.Thread(target=run_ml_service, daemon=True)
    ml_thread.start()
    
    # Wait a bit for ML service to start
    print("⏳ Waiting for ML service to initialize...")
    time.sleep(5)
    
    # Start backend
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main()
