"""
Simplified Mock Model Call Logger

This module provides a simplified implementation of the model call logger
to avoid errors in the current FastAPI backend.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "model_calls.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger("model_call_logger")

def log_model_call(
    session_id: str,
    model_id: str,
    prompt_type: str,
    prompt_query: str,
    usage_stats: Dict[str, Any],
    routing_explanation: Optional[Dict[str, Any]] = None,
    prompt_id: Optional[str] = None,
    length_category: Optional[str] = None,
    strategy: str = "balanced",
    manual_selection: bool = False,
    latency: float = 0.0,
    success: bool = True,
    error_type: Optional[str] = None,
    matched_patterns: Optional[Dict[str, int]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Simplified mock implementation of log_model_call that just logs to console/file
    """
    log_message = (
        f"Model call: {model_id} | "
        f"Type: {prompt_type} | "
        f"Strategy: {strategy} | "
        f"Tokens: {usage_stats.get('total_tokens', 0)} | "
        f"Latency: {latency:.2f}s | "
        f"Success: {success}"
    )
    
    if success:
        logger.info(log_message)
    else:
        logger.error(f"{log_message} | Error: {error_type}")
    
    # Log to console as well
    print(log_message) 