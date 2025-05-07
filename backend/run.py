#!/usr/bin/env python
"""
Run script for the FastAPI Model Router backend

This script starts the FastAPI application using uvicorn with the specified
host, port, and other settings.
"""

import os
import yaml
import logging
import uvicorn
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("runner")

def load_config():
    """Load configuration from config.yaml file"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
        return {}

def main():
    """Main entry point for running the API server"""
    # Load configuration
    config = load_config()
    server_config = config.get("server", {})
    
    # Extract server settings with defaults
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    log_level = server_config.get("log_level", "info")
    reload = server_config.get("reload", True)  # Auto-reload on code changes (development)
    
    # Set USE_MOCK environment variable if not already set
    if "USE_MOCK" not in os.environ:
        os.environ["USE_MOCK"] = "false"
        logger.info("Setting USE_MOCK=false for production responses (can be overridden in .env)")
    
    # Log startup information
    logger.info(f"Starting Model Router API on {host}:{port}")
    logger.info(f"Environment: {'development' if reload else 'production'}")
    logger.info(f"Mock mode: {'enabled' if os.environ['USE_MOCK'].lower() == 'true' else 'disabled'}")
    
    # Start the uvicorn server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )

if __name__ == "__main__":
    main() 