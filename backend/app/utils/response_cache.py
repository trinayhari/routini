import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
import time
import os
from pathlib import Path

logger = logging.getLogger("response_cache")

# Set up cache directory
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache"))
CACHE_DIR.mkdir(exist_ok=True)

# In-memory cache
memory_cache: Dict[str, Dict[str, Any]] = {}

# Cache expiration in seconds (default: 24 hours)
CACHE_EXPIRATION = int(os.environ.get("CACHE_EXPIRATION", 86400))

def generate_hash(prompt: str, model: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a SHA-256 hash from a prompt and optional parameters.
    
    Args:
        prompt: The text prompt to hash
        model: Optional model identifier
        params: Optional parameters that would affect the result (temperature, max_tokens, etc.)
    
    Returns:
        The SHA-256 hash of the prompt and parameters
    """
    # Create a dictionary with all hashable parameters
    hash_data = {
        "prompt": prompt,
    }
    
    if model:
        hash_data["model"] = model
    
    if params:
        # Only include parameters that would affect the response
        relevant_params = {
            k: v for k, v in params.items() 
            if k in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
        }
        hash_data.update(relevant_params)
    
    # Convert to string and hash
    hash_str = json.dumps(hash_data, sort_keys=True)
    hash_obj = hashlib.sha256(hash_str.encode())
    
    return hash_obj.hexdigest()

def get_cached_response(prompt_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached response from memory or disk.
    
    Args:
        prompt_hash: The SHA-256 hash of the prompt
    
    Returns:
        The cached response data or None if not found or expired
    """
    # First check in-memory cache
    if prompt_hash in memory_cache:
        cached_data = memory_cache[prompt_hash]
        
        # Check if cache has expired
        if (time.time() - cached_data.get("timestamp", 0)) > CACHE_EXPIRATION:
            logger.info(f"Cache expired for hash: {prompt_hash}")
            del memory_cache[prompt_hash]
            return None
        
        logger.info(f"Cache hit (memory) for hash: {prompt_hash}")
        return cached_data
    
    # Check disk cache
    cache_file = CACHE_DIR / f"{prompt_hash}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            
            # Check if cache has expired
            if (time.time() - cached_data.get("timestamp", 0)) > CACHE_EXPIRATION:
                logger.info(f"Cache expired for hash: {prompt_hash}")
                cache_file.unlink(missing_ok=True)
                return None
            
            # Add to memory cache for faster access next time
            memory_cache[prompt_hash] = cached_data
            
            logger.info(f"Cache hit (disk) for hash: {prompt_hash}")
            return cached_data
        except Exception as e:
            logger.warning(f"Error reading cache file: {str(e)}")
            return None
    
    return None

def cache_response(prompt_hash: str, response_data: Dict[str, Any]) -> None:
    """
    Cache a response in memory and on disk.
    
    Args:
        prompt_hash: The SHA-256 hash of the prompt
        response_data: The response data to cache
    """
    # Add timestamp to track cache age
    data_to_cache = response_data.copy()
    data_to_cache["timestamp"] = time.time()
    
    # Store in memory
    memory_cache[prompt_hash] = data_to_cache
    
    # Store on disk
    cache_file = CACHE_DIR / f"{prompt_hash}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(data_to_cache, f)
        logger.info(f"Cached response for hash: {prompt_hash}")
    except Exception as e:
        logger.warning(f"Error writing cache file: {str(e)}")

def clear_expired_cache() -> int:
    """
    Clear expired cache entries from memory and disk.
    
    Returns:
        Number of entries cleared
    """
    cleared_count = 0
    current_time = time.time()
    
    # Clear from memory
    expired_keys = [
        k for k, v in memory_cache.items() 
        if (current_time - v.get("timestamp", 0)) > CACHE_EXPIRATION
    ]
    
    for key in expired_keys:
        del memory_cache[key]
        cleared_count += 1
    
    # Clear from disk
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            
            if (current_time - cached_data.get("timestamp", 0)) > CACHE_EXPIRATION:
                cache_file.unlink()
                cleared_count += 1
        except Exception:
            # If file can't be read, assume it's corrupt and remove it
            cache_file.unlink(missing_ok=True)
            cleared_count += 1
    
    return cleared_count

def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Returns:
        Dictionary with cache statistics
    """
    # Count disk cache files
    disk_cache_files = list(CACHE_DIR.glob("*.json"))
    disk_cache_count = len(disk_cache_files)
    
    # Calculate disk cache size
    disk_cache_size = sum(f.stat().st_size for f in disk_cache_files)
    
    # Calculate memory cache size (approximate)
    memory_cache_size = sum(len(json.dumps(v)) for v in memory_cache.values())
    
    # Get some sample cache keys (up to 5)
    sample_keys = list(memory_cache.keys())[:5]
    
    # Check for expired entries
    current_time = time.time()
    expired_count = sum(
        1 for v in memory_cache.values() 
        if (current_time - v.get("timestamp", 0)) > CACHE_EXPIRATION
    )
    
    return {
        "memory_cache_count": len(memory_cache),
        "disk_cache_count": disk_cache_count,
        "memory_cache_size_bytes": memory_cache_size,
        "disk_cache_size_bytes": disk_cache_size,
        "sample_keys": sample_keys,
        "expired_count": expired_count,
        "cache_dir": str(CACHE_DIR),
        "cache_expiration_seconds": CACHE_EXPIRATION,
        "timestamp": current_time
    } 