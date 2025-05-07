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
    messages: List[Dict[str, str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = DEFAULT_TIMEOUT,
    retry_attempts: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY
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
        
    Returns:
        Tuple of (response_text, usage_stats, latency) or error dict
    """
    # Start timing for latency measurement
    start_time = time.time()
    
    # Handle both prompt and messages formats
    if messages is None:
        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either prompt or messages must be provided")
    
    # For development/testing - use mock responses
    if USE_MOCK:
        return await _generate_mock_response(model, messages, max_tokens, temperature)
    
    # Verify API key is available
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not found in environment variables")
        return {
            "error": "API key missing",
            "message": "OPENROUTER_API_KEY environment variable is not set"
        }
    
    # Prepare request headers
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bestroute.app"  # Replace with your actual domain
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
    
    # Retry loop
    while attempts < retry_attempts:
        try:
            logger.info(f"Sending request to OpenRouter API (attempt {attempts+1}/{retry_attempts})")
            
            # Use httpx for async HTTP requests with timeout
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    error_message = f"OpenRouter API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_message += f" - {json.dumps(error_data)}"
                    except:
                        error_message += f" - {response.text}"
                    
                    logger.error(error_message)
                    
                    # Determine if we should retry based on status code
                    if response.status_code in [429, 500, 502, 503, 504]:
                        attempts += 1
                        if attempts < retry_attempts:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            continue
                    
                    # If we shouldn't retry or ran out of attempts
                    return {
                        "error": "API error",
                        "message": error_message,
                        "status_code": response.status_code
                    }
                
                # Parse successful response
                try:
                    # Parse the JSON response
                    response_data = json.loads(response.text)
                    
                    # Extract the response text
                    response_text = ""
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        if "message" in response_data["choices"][0]:
                            response_text = response_data["choices"][0]["message"].get("content", "")
                        
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
                        logger.error(f"Error loading pricing from config: {e}")
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
                    
                    logger.info(f"Request successful: {model}, {total_tokens} tokens, ${total_cost:.6f}, {latency:.2f}s")
                    
                    # Return the response tuple
                    return response_text, usage_stats, latency
                    
                except Exception as e:
                    logger.error(f"Error parsing API response: {e}")
                    logger.error(f"Response content: {response.text}")
                    
                    # Try to extract just the content from the response
                    try:
                        if response.text:
                            response_text = response.text
                            return response_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0}, time.time() - start_time
                    except:
                        pass
                    
                    raise
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            last_error = e
            
            if attempts < retry_attempts - 1:
                attempts += 1
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                return {
                    "error": "Max retries exceeded",
                    "message": str(last_error)
                }
    
    return {
        "error": "Max retries exceeded",
        "message": str(last_error)
    }

async def _generate_mock_response(
    model: str, 
    messages: List[Dict[str, str]], 
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