#!/usr/bin/env python3
"""
Test script for OpenRouter API integration.
This script tests the OpenRouter API connection by sending a simple prompt.
"""

import os
import sys
import logging
import argparse
from typing import Optional

# Add the parent directory to sys.path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenRouter client
from app.utils.openrouter import send_openrouter_request

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("openrouter_test")

def test_api_connection(
    prompt: Optional[str] = None,
    model: str = "anthropic/claude-3-haiku"
) -> bool:
    """
    Test the OpenRouter API connection by sending a prompt and printing the response.
    
    Args:
        prompt: The prompt to send (default: a simple question)
        model: The model to use
        
    Returns:
        True if successful, False otherwise
    """
    # Use a default prompt if none is provided
    if prompt is None:
        prompt = "Explain in exactly one short paragraph what OpenRouter is."
    
    try:
        logger.info(f"Testing OpenRouter API with model: {model}")
        logger.info(f"Prompt: {prompt}")
        
        # Send the request
        response_text, usage_stats, latency = send_openrouter_request(
            prompt=prompt,
            model=model,
            max_tokens=200,
            temperature=0.7
        )
        
        # Print the results
        print("\n" + "="*80)
        print(f"RESPONSE FROM {model}:")
        print("="*80)
        print(response_text)
        print("="*80)
        print(f"Tokens used: {usage_stats['total_tokens']} (prompt: {usage_stats['prompt_tokens']}, completion: {usage_stats['completion_tokens']})")
        print(f"Estimated cost: ${usage_stats['cost']:.6f}")
        print(f"Response time: {latency:.2f} seconds")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    """Parse command line arguments and run the test"""
    parser = argparse.ArgumentParser(description="Test OpenRouter API integration")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the API")
    parser.add_argument("--model", type=str, default="anthropic/claude-3-haiku", 
                        help="Model to use (default: anthropic/claude-3-haiku)")
    args = parser.parse_args()
    
    success = test_api_connection(
        prompt=args.prompt,
        model=args.model
    )
    
    if success:
        print("API test completed successfully!")
        sys.exit(0)
    else:
        print("API test failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 