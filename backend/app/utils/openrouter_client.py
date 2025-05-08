#!/usr/bin/env python3
"""
OpenRouter Client

A simple client for sending requests to the OpenRouter API.
This module provides functions for sending prompts to various language models.
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

# Add the parent directory to sys.path to import from app.utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the main OpenRouter API client
from backend.app.utils.openrouter import send_request
from backend.app.utils.error_handler import with_error_handling, NetworkError, APIError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openrouter_client")

class OpenRouterClient:
    """
    A simple client for the OpenRouter API.
    
    This class provides methods for sending prompts to various language models
    via the OpenRouter API.
    """
    
    def __init__(self, default_model: str = "anthropic/claude-3-haiku"):
        """
        Initialize the OpenRouter client.
        
        Args:
            default_model: The default model to use for requests
        """
        self.default_model = default_model
    
    @with_error_handling(retries=3, delay=1.0, backoff_factor=2.0)
    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        prompt_type: Optional[str] = None,
        auto_retry_different_model: bool = True  # Add parameter to control auto-retrying with different models
    ) -> Dict[str, Any]:
        """
        Send a prompt to the OpenRouter API and get a response.
        
        Args:
            prompt: The text prompt to send
            model: The model to use (defaults to the client's default_model)
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            prompt_type: The classified prompt type (if known)
            auto_retry_different_model: Whether to automatically retry with different models on empty responses
            
        Returns:
            Dictionary containing response text, usage stats, and metrics
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")
            
        model_id = model or self.default_model
        messages = [{"role": "user", "content": prompt}]
        
        # Define fallback models in order of preference
        fallback_models = [
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "mistralai/mixtral-8x7b-instruct",
            "meta-llama/llama-2-70b-chat"
        ]
        # Remove the current model from fallbacks if it's in the list
        if model_id in fallback_models:
            fallback_models.remove(model_id)
        
        # Keep track of tried models to avoid infinite loop
        tried_models = [model_id]
        current_model = model_id
        
        for attempt in range(3):  # Limit total attempts to 3
            start_time = time.time()
            logger.info(f"Attempt {attempt+1}: Sending prompt to OpenRouter API using model: {current_model}")
            
            response = await send_request(
                model=current_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Handle API errors
            if isinstance(response, dict) and "error" in response:
                error_msg = response.get("message", "Unknown error") 
                status_code = response.get("status_code", 500)
                logger.error(f"OpenRouter API error: {error_msg} (Status: {status_code})")
                
                # Convert to the appropriate error type for our error handling system
                if status_code >= 500:
                    raise NetworkError(f"Server error: {error_msg}", status_code=status_code)
                else:
                    raise APIError(f"API error: {error_msg}", status_code=status_code)
            
            response_text, usage_stats, latency = response
            
            # Check if we got a valid response
            is_empty_response = not response_text or response_text.strip() == ""
            contains_error_message = any(error_pattern in response_text.lower() 
                                       for error_pattern in ["empty response", "content filtering", "rate limit"])
            
            if is_empty_response or contains_error_message:
                logger.warning(f"Model {current_model} returned empty/error response after {latency:.2f}s")
                
                # Only try other models if auto_retry_different_model is enabled and we haven't exhausted all fallbacks
                if auto_retry_different_model and fallback_models and attempt < 2:
                    # Get the next fallback model
                    next_model = fallback_models.pop(0)
                    tried_models.append(next_model)
                    current_model = next_model
                    logger.info(f"Switching to fallback model: {current_model}")
                    # Continue to the next attempt with the new model
                    continue
                else:
                    # If auto-retry is disabled or we've tried all models, return a helpful message
                    response_text = (
                        f"I apologize, but I'm having trouble generating a response at the moment. "
                        f"This could be due to temporary service issues. "
                        f"Please try rephrasing your question or try again later. "
                        f"(I tried these models: {', '.join(tried_models)})"
                    )
            else:
                # We got a valid response, no need to try more models
                logger.info(f"Got valid response from {current_model}: {len(response_text)} chars, "
                           f"{usage_stats['total_tokens']} tokens in {latency:.2f}s")
                break
        
        return {
            "success": True,
            "text": response_text,
            "model": current_model,
            "tokens": usage_stats["total_tokens"],
            "prompt_tokens": usage_stats["prompt_tokens"],
            "completion_tokens": usage_stats["completion_tokens"],
            "cost": usage_stats["cost"],
            "latency": latency,
            "prompt_type": prompt_type,
            "tried_models": tried_models
        }
    
    @with_error_handling(retries=3, delay=1.0, backoff_factor=2.0)
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        prompt_type: Optional[str] = None,
        auto_retry_different_model: bool = True  # Add parameter to control auto-retrying with different models
    ) -> Dict[str, Any]:
        """
        Send a chat conversation to the OpenRouter API.
        
        Args:
            messages: List of message objects (each with 'role' and 'content')
            model: The model to use (defaults to the client's default_model)
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            prompt_type: The classified prompt type (if known)
            auto_retry_different_model: Whether to automatically retry with different models on empty responses
            
        Returns:
            Dictionary containing response text, usage stats, and metrics
        """
        if not messages or len(messages) == 0:
            raise ValueError("Messages list cannot be empty")
            
        # Ensure all messages have content
        for i, msg in enumerate(messages):
            if not msg.get("content") or msg.get("content").strip() == "":
                raise ValueError(f"Message at index {i} has empty content")
        
        model_id = model or self.default_model
        
        # Define fallback models in order of preference
        fallback_models = [
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "mistralai/mixtral-8x7b-instruct",
            "meta-llama/llama-2-70b-chat"
        ]
        # Remove the current model from fallbacks if it's in the list
        if model_id in fallback_models:
            fallback_models.remove(model_id)
        
        # Keep track of tried models to avoid infinite loop
        tried_models = [model_id]
        current_model = model_id
        
        for attempt in range(3):  # Limit total attempts to 3
            start_time = time.time()
            logger.info(f"Attempt {attempt+1}: Sending chat to OpenRouter API using model: {current_model}")
            
            response = await send_request(
                model=current_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Handle API errors
            if isinstance(response, dict) and "error" in response:
                error_msg = response.get("message", "Unknown error")
                status_code = response.get("status_code", 500) 
                logger.error(f"OpenRouter API error: {error_msg} (Status: {status_code})")
                
                # Convert to the appropriate error type for our error handling system
                if status_code >= 500:
                    raise NetworkError(f"Server error: {error_msg}", status_code=status_code)
                else:
                    raise APIError(f"API error: {error_msg}", status_code=status_code)
            
            response_text, usage_stats, latency = response
            
            # Check if we got a valid response
            is_empty_response = not response_text or response_text.strip() == ""
            contains_error_message = any(error_pattern in response_text.lower() 
                                       for error_pattern in ["empty response", "content filtering", "rate limit"])
            
            if is_empty_response or contains_error_message:
                logger.warning(f"Model {current_model} returned empty/error response after {latency:.2f}s")
                
                # Only try other models if auto_retry_different_model is enabled and we haven't exhausted all fallbacks
                if auto_retry_different_model and fallback_models and attempt < 2:
                    # Get the next fallback model
                    next_model = fallback_models.pop(0)
                    tried_models.append(next_model)
                    current_model = next_model
                    logger.info(f"Switching to fallback model: {current_model}")
                    # Continue to the next attempt with the new model
                    continue
                else:
                    # If auto-retry is disabled or we've tried all models, return a helpful message
                    response_text = (
                        f"I apologize, but I'm having trouble generating a response at the moment. "
                        f"This could be due to temporary service issues. "
                        f"Please try rephrasing your question or try again later. "
                        f"(I tried these models: {', '.join(tried_models)})"
                    )
            else:
                # We got a valid response, no need to try more models
                logger.info(f"Got valid response from {current_model}: {len(response_text)} chars, "
                           f"{usage_stats['total_tokens']} tokens in {latency:.2f}s")
                break
        
        return {
            "success": True,
            "text": response_text,
            "model": current_model,
            "tokens": usage_stats["total_tokens"],
            "prompt_tokens": usage_stats["prompt_tokens"],
            "completion_tokens": usage_stats["completion_tokens"],
            "cost": usage_stats["cost"],
            "latency": latency,
            "prompt_type": prompt_type,
            "tried_models": tried_models
        }
    
    async def compare(
        self,
        prompt: str,
        models: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Compare responses from multiple models for the same prompt.
        
        Args:
            prompt: The text prompt to send
            models: List of model IDs to compare
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary with results from each model
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")
            
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        logger.info(f"Comparing {len(models)} models for prompt: {prompt[:50]}...")
        
        # Query each model in parallel
        @with_error_handling(retries=2, delay=1.0, backoff_factor=1.5)
        async def query_model(model):
            model_start = time.time()
            
            response = await send_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if isinstance(response, dict) and "error" in response:
                error_msg = response.get("message", "Unknown error")
                status_code = response.get("status_code", 500)
                logger.error(f"OpenRouter API error for model {model}: {error_msg} (Status: {status_code})")
                
                # Convert to the appropriate error type for our error handling system
                if status_code >= 500:
                    raise NetworkError(f"Server error: {error_msg}", status_code=status_code)
                else:
                    raise APIError(f"API error: {error_msg}", status_code=status_code)
            
            response_text, usage_stats, _ = response
            
            # Check for empty responses
            if not response_text or response_text.strip() == "":
                logger.warning(f"Received empty response from model: {model}")
                # Check if this is the default error message pattern or something else
                if response_text.startswith("The model returned an empty response"):
                    logger.info(f"Using the detailed error message from the API")
                else:
                    # If we don't have a detailed message, provide a standard one
                    response_text = (
                        f"The {model} model returned an empty response. "
                        f"This may be due to content filtering, rate limiting, or a model issue. "
                        f"Try rephrasing your prompt or selecting a different model."
                    )
            
            return {
                "success": True,
                "text": response_text,
                "model": model,
                "tokens": usage_stats["total_tokens"],
                "prompt_tokens": usage_stats["prompt_tokens"],
                "completion_tokens": usage_stats["completion_tokens"],
                "cost": usage_stats["cost"],
                "latency": time.time() - model_start
            }
        
        # Create and run tasks for all models
        tasks = [query_model(model) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Filter out any failed results and replace with error messages
        processed_results = []
        for i, (model, result) in enumerate(zip(models, results)):
            if isinstance(result, dict) and "success" in result:
                processed_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Model {model} comparison failed: {str(result)}")
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "model": model,
                    "latency": 0
                })
            else:
                logger.error(f"Unexpected result type for model {model}: {type(result)}")
                processed_results.append({
                    "success": False, 
                    "error": "Unexpected error",
                    "model": model,
                    "latency": 0
                })
        
        logger.info(f"Comparison completed in {time.time() - start_time:.2f}s")
        
        return {
            "prompt": prompt,
            "results": processed_results,
            "total_latency": time.time() - start_time
        }

# Create a singleton instance for easy importing
client = OpenRouterClient()

# -------------------------------------------------------------------------------
# Convenience functions for simpler usage
# -------------------------------------------------------------------------------

async def ask(
    prompt: str,
    model: str = "anthropic/claude-3-haiku",
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Send a prompt to the OpenRouter API and get a response.
    
    Args:
        prompt: The text prompt to send
        model: The model to use
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Dictionary containing response text and metrics
    """
    return await client.ask(prompt, model, max_tokens, temperature)

async def chat(
    messages: List[Dict[str, str]],
    model: str = "anthropic/claude-3-haiku",
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Send a chat conversation to the OpenRouter API.
    
    Args:
        messages: List of message objects (each with 'role' and 'content')
        model: The model to use
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Dictionary containing response text and metrics
    """
    return await client.chat(messages, model, max_tokens, temperature)

async def compare(
    prompt: str,
    models: List[str],
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Compare responses from multiple models for the same prompt.
    
    Args:
        prompt: The text prompt to send
        models: List of model IDs to compare
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Dictionary with results from each model
    """
    return await client.compare(prompt, models, max_tokens, temperature)

# -------------------------------------------------------------------------------
# Simple example usage
# -------------------------------------------------------------------------------

async def main():
    """Run a simple example"""
    # Ask a question
    response = await ask("What is the capital of France?")
    
    if response["success"]:
        print(f"Response: {response['text']}")
        print(f"Tokens: {response['tokens']} | Cost: ${response['cost']:.6f} | Latency: {response['latency']:.2f}s")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Compare models
    comparison = await compare(
        "Explain quantum computing in simple terms.",
        ["anthropic/claude-3-haiku", "openai/gpt-4"]
    )
    
    print("\nModel Comparison:")
    for result in comparison["results"]:
        model = result["model"]
        print(f"\n--- {model} ---")
        
        if result["success"]:
            print(f"Response (first 100 chars): {result['text'][:100]}...")
            print(f"Tokens: {result['tokens']} | Cost: ${result['cost']:.6f} | Latency: {result['latency']:.2f}s")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 