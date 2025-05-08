"""
OpenRouter API client for sending requests to different language models.

This module provides functionality to send prompts to the OpenRouter API,
which routes requests to various language models.
"""

import os
import time
import json
import httpx
import asyncio
import logging
import yaml
import tiktoken
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv
from random import uniform

# Import error handling
from .error_handler import APIError, NetworkError, AuthenticationError, RateLimitError, ModelError, UnknownAPIError, classify_http_error

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openrouter_client")

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
BACKOFF_FACTOR = 1.5  # exponential backoff multiplier
# Use environment variable to control mock mode, default to false if not set
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

# Model to encoding mapping for tiktoken
MODEL_ENCODINGS = {
    "openai/gpt-4": "cl100k_base",  # GPT-4 uses cl100k_base
    "openai/gpt-3.5-turbo": "cl100k_base",  # GPT-3.5 uses cl100k_base
    "anthropic/claude-3-opus": "cl100k_base",  # Claude 3 uses cl100k_base
    "anthropic/claude-3-sonnet": "cl100k_base",
    "anthropic/claude-3-haiku": "cl100k_base",
    "mistralai/mixtral-8x7b-instruct": "cl100k_base",  # Mixtral uses cl100k_base
    "mistralai/mistral-7b-instruct": "cl100k_base",
    "meta-llama/llama-2-70b-chat": "cl100k_base",  # Llama 2 uses cl100k_base
}

# Add a global client with a session pool for better connection reuse
_http_client = None

# Track request count to recreate the client periodically
_request_count = 0
_max_requests_per_client = 5  # Recreate client after this many requests

async def get_http_client():
    """Get or create a shared HTTP client with a connection pool"""
    global _http_client, _request_count
    
    # Create a new client if needed
    if _http_client is None or _request_count >= _max_requests_per_client:
        # Close existing client if it exists
        if _http_client is not None:
            logger.info(f"Closing and recreating HTTP client after {_request_count} requests")
            await _http_client.aclose()
        
        # Create new client
        _http_client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        _request_count = 0
    
    # Increment request count
    _request_count += 1
    
    return _http_client

# Add a tracking variable for last request time to help with rate limits
_last_request_time = 0

def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens in a text using tiktoken.
    Falls back to word count if model encoding is not supported.
    """
    try:
        encoding_name = MODEL_ENCODINGS.get(model)
        if encoding_name:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        else:
            # Fallback to word count for unsupported models
            return len(text.split())
    except Exception as e:
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Fallback to word count
        return len(text.split())

# For capital of France prompt, customize mock responses to be more realistic
CAPITAL_FRANCE_RESPONSES = {
    "openai/gpt-4": "The capital of France is Paris. It's the largest city in France and one of the most iconic cities in the world, known for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
    "anthropic/claude-3-opus": "The capital of France is Paris.\n\nParis is not only the political capital but also the cultural and economic center of France. It's located in the north-central part of the country on the Seine River. Paris is famous for its art, architecture, cuisine, and landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
    "anthropic/claude-3-sonnet": "The capital of France is Paris. It is located in the north-central part of the country on the Seine River. Paris is known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
    "anthropic/claude-3-haiku": "The capital of France is Paris.",
    "mistralai/mixtral-8x7b-instruct": "Paris is the capital city of France. It is located in the north-central part of the country, on the Seine River.",
    "meta-llama/llama-2-70b-chat": "The capital of France is Paris. It's situated in the north-central part of the country on the Seine River."
}

async def send_request(
    model: str,
    prompt: str = None,
    messages: List[Dict[str, Any]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = DEFAULT_TIMEOUT,
    retry_attempts: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY,
    debug_mode: bool = False
) -> Union[Tuple[str, Dict[str, Any], float], Dict[str, Any]]:
    """
    Send a request to the OpenRouter API with retry logic and timeout handling.
    
    Args:
        model: The model ID to use (e.g., "openai/gpt-4")
        prompt: The prompt text (deprecated, use messages instead)
        messages: List of message objects (each with 'role' and 'content')
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        timeout: Request timeout in seconds
        retry_attempts: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        debug_mode: Whether to log full response contents for debugging
        
    Returns:
        Tuple of (response_text, usage_stats, latency) or error dict
    """
    global _last_request_time
    
    # Start timing for latency measurement
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    # Handle both prompt and messages formats
    if messages is None:
        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        else:
            error_msg = "Either prompt or messages must be provided"
            logger.error(f"[{request_id}] {error_msg}")
            raise ValueError(error_msg)
    
    # Validate that messages are non-empty
    for i, msg in enumerate(messages):
        if not msg.get("content") or msg.get("content").strip() == "":
            error_msg = f"Message at index {i} has empty content"
            logger.error(f"[{request_id}] {error_msg}")
            raise ValueError(error_msg)
    
    # For development/testing - use mock responses
    if USE_MOCK:
        return await _generate_mock_response(model, messages, max_tokens, temperature)
    
    # Verify API key is available
    if not OPENROUTER_API_KEY:
        logger.error(f"[{request_id}] OpenRouter API key not found in environment variables")
        raise AuthenticationError("OPENROUTER_API_KEY environment variable is not set")
    
    # Add jitter to avoid rate limits by ensuring minimum time between requests
    # Calculate time since last request
    now = time.time()
    time_since_last_request = now - _last_request_time
    
    # We want at least 500ms between requests to avoid rate limiting
    # Add some jitter (100-300ms) to avoid request patterns
    if time_since_last_request < 0.5 and _last_request_time > 0:
        delay_needed = 0.5 - time_since_last_request + uniform(0.1, 0.3)
        logger.info(f"[{request_id}] Adding {delay_needed:.2f}s delay to avoid rate limits")
        await asyncio.sleep(delay_needed)
    
    # Update last request time
    _last_request_time = time.time()
    
    # Prepare request headers with a randomized user agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bestroute.app",  # Replace with your actual domain
        "User-Agent": user_agents[int(time.time()) % len(user_agents)]
    }
    
    # Add a unique request ID to help with tracing
    request_headers = {
        **headers,
        "X-Request-ID": request_id
    }
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Initialize variables for retry logic
    attempts = 0
    last_error = None
    
    # Get shared HTTP client
    client = await get_http_client()
    
    # Retry loop
    while attempts <= retry_attempts:
        try:
            logger.info(f"[{request_id}] Sending request to OpenRouter API (attempt {attempts+1}/{retry_attempts+1})")
            
            response = await client.post(
                OPENROUTER_API_URL,
                headers=request_headers,
                json=payload,
                timeout=timeout
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_message = f"OpenRouter API error: {response.status_code}"
                error_data = {}
                
                try:
                    error_data = response.json()
                    error_message += f" - {json.dumps(error_data)}"
                except:
                    error_message += f" - {response.text}"
                
                logger.error(f"[{request_id}] {error_message}")
                
                # Use our error classification to create the appropriate exception
                api_error = classify_http_error(response.status_code, error_data)
                
                # Determine if we should retry based on error type
                if isinstance(api_error, (NetworkError, RateLimitError)) and attempts < retry_attempts:
                    attempts += 1
                    # Add increasing delay with jitter for rate limits
                    actual_delay = retry_delay * (attempts * BACKOFF_FACTOR) + uniform(0.1, 0.5)
                    logger.info(f"[{request_id}] Retrying in {actual_delay:.2f} seconds...")
                    await asyncio.sleep(actual_delay)
                    continue
                
                # If we shouldn't retry or ran out of attempts, raise the error
                raise api_error
            
            # Parse successful response
            try:
                # Parse the JSON response
                response_data = json.loads(response.text)
                
                # Log full response in debug mode
                if debug_mode:
                    logger.info(f"[{request_id}] Full response: {json.dumps(response_data)}")
                
                # Extract the response text
                response_text = ""
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    if "message" in response_data["choices"][0]:
                        response_text = response_data["choices"][0]["message"].get("content", "")
                
                # Check finish reason for better error handling
                finish_reason = None
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    finish_reason = response_data["choices"][0].get("finish_reason", None)
                
                # If we got an empty response, this is abnormal but we'll provide a message
                if not response_text or response_text.strip() == "":
                    logger.warning(f"[{request_id}] Received empty response text from model: {model}")
                    
                    # Try to extract any useful information from the response for troubleshooting
                    model_used = response_data.get("model", model)
                    
                    # Log detailed info about the empty response
                    logger.error(f"[{request_id}] Empty response details - Model: {model_used}, " +
                                f"Finish reason: {finish_reason}, " +
                                f"Response structure: {json.dumps(response_data)[:200]}...")
                    
                    # Check if we should retry - retry if we have attempts left and either:
                    # 1. Finish reason is 'length' (context length issue)
                    # 2. Finish reason is 'content_filter' (content was filtered)
                    if attempts < retry_attempts and finish_reason in ["length", "content_filter"]:
                        attempts += 1
                        logger.info(f"[{request_id}] Retrying due to {finish_reason} (attempt {attempts}/{retry_attempts})")
                        await asyncio.sleep(retry_delay * attempts)
                        continue
                    
                    # Detailed error message with troubleshooting information
                    response_text = (
                        f"The model returned an empty response. This could be due to content filtering, "
                        f"rate limiting, or a temporary issue with the model. "
                        f"Finish reason: {finish_reason}. "
                        f"Try rephrasing your prompt or using a different model."
                    )
                
                # Extract token usage information
                usage = response_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Load config for pricing
                try:
                    with open("config.yaml", "r") as f:
                        config = yaml.safe_load(f)
                        model_info = config.get("models", {}).get(model, {})
                        pricing = model_info.get("pricing", {})
                        input_cost = pricing.get("input", 0.0)  # per 1K tokens
                        output_cost = pricing.get("output", 0.0)  # per 1K tokens
                        
                        # Calculate total cost
                        total_cost = (
                            (prompt_tokens * input_cost / 1000) +
                            (completion_tokens * output_cost / 1000)
                        )
                except Exception as e:
                    logger.error(f"[{request_id}] Error loading pricing from config: {e}")
                    total_cost = 0.0
                
                # Calculate total latency
                latency = time.time() - start_time
                
                # Prepare usage statistics
                usage_stats = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": total_cost
                }
                
                logger.info(f"[{request_id}] Request successful: {model}, {total_tokens} tokens, ${total_cost:.6f}, {latency:.2f}s")
                
                # Return the response tuple
                return response_text, usage_stats, latency
                
            except Exception as e:
                logger.error(f"[{request_id}] Error parsing API response: {e}")
                logger.error(f"[{request_id}] Response content: {response.text}")
                
                # Try to extract just the content from the response
                try:
                    if response.text:
                        # Try to parse as JSON first
                        try:
                            data = json.loads(response.text)
                            if "choices" in data and len(data["choices"]) > 0:
                                if "message" in data["choices"][0]:
                                    response_text = data["choices"][0]["message"].get("content", "")
                                    if response_text:
                                        logger.info(f"[{request_id}] Successfully extracted text from partial response")
                                        return response_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0}, time.time() - start_time
                        except:
                            # If not JSON, use the raw text
                            response_text = response.text
                            return response_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0}, time.time() - start_time
                except:
                    pass
                
                # If we couldn't extract anything useful, raise the original error
                raise
        
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            # Network errors
            logger.error(f"[{request_id}] Network error: {str(e)}")
            last_error = NetworkError(f"Connection error: {str(e)}")
        
        except (APIError, NetworkError, AuthenticationError, RateLimitError, ModelError) as e:
            # These are already our custom errors, so just log and possibly retry
            logger.error(f"[{request_id}] API error: {str(e)}")
            last_error = e
        
        except Exception as e:
            # Unexpected errors
            logger.error(f"[{request_id}] Unexpected error: {str(e)}")
            last_error = UnknownAPIError(f"Unexpected error: {str(e)}")
        
        # Increment attempts and retry
        attempts += 1
        if attempts <= retry_attempts:
            logger.info(f"[{request_id}] Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
        else:
            # If we've exhausted all retries, return an error response
            logger.error(f"[{request_id}] Max retries exceeded. Last error: {str(last_error)}")
            return {
                "error": "Max retries exceeded",
                "message": str(last_error),
                "status_code": getattr(last_error, "status_code", 500)
            }
    
    # This should never happen if the retry loop is working correctly
    return {
        "error": "Max retries exceeded",
        "message": str(last_error) if last_error else "Unknown error",
        "status_code": getattr(last_error, "status_code", 500)
    }

async def _generate_mock_response(
    model: str, 
    messages: List[Dict[str, Any]], 
    max_tokens: int,
    temperature: float
) -> Tuple[str, Dict[str, Any], float]:
    """
    Generate a mock response for development/testing purposes.
    
    Args:
        model: The model ID
        messages: List of message objects
        max_tokens: Maximum tokens to generate
        temperature: Temperature setting
        
    Returns:
        Tuple of (response_text, usage_stats, latency)
    """
    # Extract the last user message as the prompt
    prompt = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            prompt = message.get("content", "")
            break
    
    # Simulate network delay and processing time based on model
    model_latency = {
        "openai/gpt-4": 2.5,
        "anthropic/claude-3-opus": 3.5,
        "anthropic/claude-3-sonnet": 2.0,
        "anthropic/claude-3-haiku": 1.0,
        "mistralai/mixtral-8x7b-instruct": 1.2,
        "meta-llama/llama-2-70b-chat": 1.8
    }.get(model, 2.0)
    
    # Add some randomness to the latency
    import random
    latency = model_latency * random.uniform(0.8, 1.2)
    
    # Simulate processing time
    await asyncio.sleep(latency)
    
    # Determine prompt type for better mock responses
    is_coding_prompt = any(keyword in prompt.lower() for keyword in 
                         ["code", "function", "program", "algorithm", "python", "javascript", "write a"])
    is_question = any(prompt.lower().startswith(starter) for starter in 
                    ["what", "who", "where", "when", "why", "how", "is", "are", "can", "does", "do"]) or "?" in prompt
    is_summary = any(keyword in prompt.lower() for keyword in 
                   ["summarize", "summary", "tldr", "recap", "outline"])
    
    # Customized responses for specific prompts
    if "capital of france" in prompt.lower():
        response_text = CAPITAL_FRANCE_RESPONSES.get(model, "The capital of France is Paris.")
    elif "is it working" in prompt.lower() or "does this work" in prompt.lower():
        response_text = f"Yes, the API is working correctly! This is a response from the {model} model. This confirms that your request was processed successfully and the model router is functioning as expected."
    elif "fibonacci" in prompt.lower():
        if "openai/gpt-4" in model:
            response_text = """The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1.

Here's a Python function to generate Fibonacci numbers:

```python
def fibonacci(n):
    # Generate the first n Fibonacci numbers
    fib_sequence = [0, 1]
    
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return fib_sequence
    
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    return fib_sequence

# Example usage
print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

The sequence starts: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, etc."""
        else:
            response_text = "The Fibonacci sequence is 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...\nEach number is the sum of the two preceding ones."
    elif "weather" in prompt.lower():
        response_text = "I don't have access to real-time weather data. To get current weather information, you should check a weather service or website like weather.gov or accuweather.com."
    elif is_coding_prompt:
        if "python" in prompt.lower() and "fibonacci" in prompt.lower():
            response_text = """```python
def fibonacci(n):
    # Calculate the nth Fibonacci number efficiently using dynamic programming
    if n <= 0:
        return 0
    elif n == 1:
        return 1
        
    # Initialize array to store Fibonacci numbers
    fib = [0] * (n + 1)
    fib[1] = 1
    
    # Bottom-up approach to fill the array
    for i in range(2, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
    
    return fib[n]

# Example usage
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")
```

This implementation uses dynamic programming to efficiently calculate Fibonacci numbers, avoiding the exponential complexity of a naive recursive approach."""
        elif "javascript" in prompt.lower():
            response_text = """Here's a solution in JavaScript:

```javascript
function createCounter() {
  let count = 0;
  
  return {
    increment: function() {
      return ++count;
    },
    decrement: function() {
      return --count;
    },
    getValue: function() {
      return count;
    },
    reset: function() {
      count = 0;
      return count;
    }
  };
}

// Example usage
const counter = createCounter();
console.log(counter.getValue()); // 0
console.log(counter.increment()); // 1
console.log(counter.increment()); // 2
console.log(counter.decrement()); // 1
console.log(counter.reset());     // 0
```

This implementation uses a closure to maintain private state while exposing only the methods needed to interact with the counter."""
        else:
            response_text = f"Based on your request, here's a sample implementation that should help. Note that this is a simulated response from {model}, so you might want to adapt it to your specific requirements."
    elif is_question:
        provider = model.split('/')[0]
        if provider == "openai":
            response_text = f"According to my knowledge as a large language model, I can provide information about your question. However, as this is a simulated {model} response for testing purposes, I'm not generating a complete answer. In a production environment, you would receive a detailed response addressing your specific question."
        elif provider == "anthropic":
            response_text = f"This is a simulated response from {model}. In a real implementation, Claude would analyze your question and provide a thoughtful, comprehensive answer based on its training data. The response would be balanced, nuanced, and would indicate any limitations or uncertainties in the information provided."
        else:
            response_text = f"This is a simulated response from {model}. In a production environment, this model would provide an informative answer to your question, drawing on its training data to give you accurate and helpful information."
    else:
        # Generic model-specific responses
        if "gpt" in model:
            response_text = f"This is a simulated response from {model}. In a real implementation, this would contain a detailed, informative answer generated by OpenAI's model based on your prompt. The response would be well-structured and would address your request comprehensively."
        elif "claude" in model:
            response_text = f"As a simulated {model} response, I would normally provide a thoughtful and nuanced answer to your query. Claude models are known for their helpfulness, harmlessness, and honesty in addressing user requests. This is just a placeholder for development purposes."
        elif "mixtral" in model:
            response_text = f"[Mixtral simulated response] This would typically be a well-reasoned response to your input. Mixtral models are known for their strong performance across diverse tasks with efficient computation. This is just a mock response during development."
        else:
            response_text = f"This is a simulated response from {model}. In a production environment, you would receive an actual model-generated response addressing your specific request."
    
    # Count tokens using tiktoken
    prompt_tokens = count_tokens(prompt, model)
    completion_tokens = count_tokens(response_text, model)
    total_tokens = prompt_tokens + completion_tokens
    
    # Load pricing from config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            model_info = config.get("models", {}).get(model, {})
            pricing = model_info.get("pricing", {})
            input_cost = pricing.get("input", 0.0)  # per 1K tokens
            output_cost = pricing.get("output", 0.0)  # per 1K tokens
            
            # Calculate total cost
            total_cost = (
                (prompt_tokens * input_cost / 1000) +
                (completion_tokens * output_cost / 1000)
            )
    except Exception as e:
        logger.error(f"Error loading pricing from config: {e}")
        total_cost = 0.0
    
    # Prepare usage statistics
    usage_stats = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": total_cost
    }
    
    logger.info(f"Generated mock response: {model}, {total_tokens} tokens, ${total_cost:.6f}, {latency:.2f}s")
    
    return response_text, usage_stats, latency 

def send_openrouter_request(
    prompt: str,
    model: str = "anthropic/claude-3-haiku",
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Tuple[str, Dict[str, Any], float]:
    """
    Simplified function to send a prompt to OpenRouter API and get response with metrics.
    
    Args:
        prompt: The text prompt to send to the model
        model: The model ID to use (e.g., "anthropic/claude-3-sonnet")
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Tuple of (response_text, usage_stats, latency_seconds)
        
    Raises:
        Exception: If the request fails or returns an error
    """
    logger.info(f"Sending prompt to OpenRouter API using model: {model}")
    
    # Create a simple message format
    messages = [{"role": "user", "content": prompt}]
    
    # Call the main async function
    try:
        # Handle asyncio properly - check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, use it
                response = loop.run_until_complete(send_request(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                ))
            else:
                # If no loop is running, create a new one
                response = asyncio.run(send_request(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                ))
        except RuntimeError:
            # If we can't get a loop, create a new one
            response = asyncio.run(send_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            ))
        
        # Check if we got an error dictionary
        if isinstance(response, dict) and "error" in response:
            error_msg = response.get("message", "Unknown error")
            logger.error(f"OpenRouter API error: {error_msg}")
            raise Exception(f"OpenRouter API request failed: {error_msg}")
            
        # Unpack the successful response tuple
        response_text, usage_stats, latency = response
        
        logger.info(f"OpenRouter API request successful: {len(response_text)} chars, " +
                   f"{usage_stats['total_tokens']} tokens, {latency:.2f}s")
        
        return response_text, usage_stats, latency
        
    except Exception as e:
        logger.error(f"Error in send_openrouter_request: {str(e)}")
        raise

# Add this function near the end of the file, before test_openrouter()
async def diagnose_empty_response(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Run a diagnostic test on a prompt and model to analyze empty response issues.
    
    Args:
        model: The model to test
        prompt: The prompt to test
        max_tokens: Maximum tokens to generate
        temperature: Temperature setting
        
    Returns:
        Dictionary with diagnostic results
    """
    logger.info(f"Running empty response diagnostic on model: {model}")
    
    start_time = time.time()
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = await send_request(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_mode=True  # Enable detailed debug information
        )
        
        if isinstance(response, dict) and "error" in response:
            return {
                "success": False,
                "model": model,
                "error": response.get("message", "Unknown error"),
                "status_code": response.get("status_code", 500),
                "latency": time.time() - start_time
            }
        
        response_text, usage_stats, latency = response
        
        # Check if response is empty
        is_empty_response = not response_text or response_text.strip() == ""
        
        # Check if response contains an error message about empty responses
        error_patterns = [
            "empty response",
            "no response", 
            "content filtering",
            "content policy",
            "content moderation",
            "rate limit",
            "try again",
            "not available"
        ]
        
        contains_error_message = any(pattern in response_text.lower() for pattern in error_patterns)
        
        return {
            "success": True,
            "model": model,
            "response_text": response_text,
            "usage_stats": usage_stats,
            "latency": latency,
            "is_empty_response": is_empty_response,
            "contains_error_message": contains_error_message,
            "total_time": time.time() - start_time
        }
    
    except Exception as e:
        return {
            "success": False,
            "model": model,
            "error": str(e),
            "latency": time.time() - start_time
        }

# Add diagnostic command to the test function
def test_openrouter():
    """Test the OpenRouter API with a simple prompt"""
    try:
        # Regular test
        response, usage, latency = send_openrouter_request(
            prompt="What is the capital of France?",
            model="anthropic/claude-3-haiku"
        )
        
        print(f"Response: {response[:100]}...")
        print(f"Tokens: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
        print(f"Cost: ${usage['cost']:.6f}")
        print(f"Latency: {latency:.2f}s")
        
        # Adding empty response diagnosis if required
        if "empty response" in response:
            print("\nRunning diagnostic test for empty responses...")
            
            # Handle asyncio properly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    diagnostic = loop.run_until_complete(diagnose_empty_response(
                        model="anthropic/claude-3-opus",
                        prompt="Hello, how are you today?"
                    ))
                else:
                    diagnostic = asyncio.run(diagnose_empty_response(
                        model="anthropic/claude-3-opus",
                        prompt="Hello, how are you today?"
                    ))
            except RuntimeError:
                diagnostic = asyncio.run(diagnose_empty_response(
                    model="anthropic/claude-3-opus",
                    prompt="Hello, how are you today?"
                ))
                
            print(f"Diagnostic results: {json.dumps(diagnostic, indent=2)}")
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # If this file is run directly, test the OpenRouter API
    test_openrouter() 