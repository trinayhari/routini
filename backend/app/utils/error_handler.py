"""
Error handling utilities for API requests.

This module provides standardized error handling with retry logic 
for API requests to external services like OpenRouter.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union, Awaitable
from functools import wraps

# Set up logging
logger = logging.getLogger("error_handler")

# Type for the wrapped function's return value
T = TypeVar('T')

class APIError(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class NetworkError(APIError):
    """Exception for network-related errors (connection issues, timeouts)"""
    pass

class AuthenticationError(APIError):
    """Exception for authentication failures (invalid API keys)"""
    pass

class RateLimitError(APIError):
    """Exception for rate limiting by the API provider"""
    pass

class ModelError(APIError):
    """Exception for model-specific errors (context too long, prompt rejected)"""
    pass

class UnknownAPIError(APIError):
    """Exception for unexpected API errors"""
    pass

async def async_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: tuple = (NetworkError, RateLimitError),
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: The async function to retry
        *args: Positional arguments to pass to the function
        retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay between retries
        retry_on: Tuple of exception types to retry on
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The return value of the function if successful
        
    Raises:
        The last exception if all retries fail
    """
    last_error = None
    current_delay = delay
    
    # Try the initial call plus retries
    for attempt in range(retries + 1):
        try:
            return await func(*args, **kwargs)
            
        except retry_on as e:
            last_error = e
            
            # Don't sleep on the last attempt
            if attempt < retries:
                # Log the error and retry info
                logger.warning(
                    f"API call failed (attempt {attempt+1}/{retries+1}): {str(e)}. "
                    f"Retrying in {current_delay:.2f}s..."
                )
                
                # Wait before retrying
                await asyncio.sleep(current_delay)
                
                # Increase the delay for the next attempt
                current_delay *= backoff_factor
            else:
                # Log the final failure
                logger.error(
                    f"API call failed after {retries+1} attempts: {str(e)}. "
                    f"Giving up."
                )
                
        except Exception as e:
            # For other exceptions, don't retry
            logger.error(f"API call failed with non-retryable error: {str(e)}")
            raise
    
    # If we get here, all retries failed
    if last_error:
        raise last_error
    
    # This should never happen if the function raises an exception on failure
    raise UnknownAPIError("All retries failed with an unknown error")

def with_error_handling(
    retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Decorator to add error handling and retry logic to async functions.
    
    Args:
        retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay between retries
        
    Returns:
        Decorated function with error handling and retry logic
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = f"req_{int(time.time())}"
            start_time = time.time()
            
            logger.info(f"[{request_id}] Starting API call to {func.__name__}")
            
            try:
                # Use the retry logic
                result = await async_retry(
                    func, 
                    *args, 
                    retries=retries,
                    delay=delay,
                    backoff_factor=backoff_factor,
                    **kwargs
                )
                
                # Log successful completion
                elapsed = time.time() - start_time
                logger.info(f"[{request_id}] API call to {func.__name__} completed successfully in {elapsed:.3f}s")
                
                return result
                
            except NetworkError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] Network error after {elapsed:.3f}s: {str(e)}")
                return {
                    "success": False,
                    "error": f"Network error: {str(e)}",
                    "error_type": "network",
                    "status_code": getattr(e, "status_code", None),
                    "latency": elapsed
                }
                
            except AuthenticationError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] Authentication error after {elapsed:.3f}s: {str(e)}")
                return {
                    "success": False,
                    "error": f"Authentication error: {str(e)}",
                    "error_type": "auth",
                    "status_code": getattr(e, "status_code", None),
                    "latency": elapsed
                }
                
            except RateLimitError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] Rate limit error after {elapsed:.3f}s: {str(e)}")
                return {
                    "success": False,
                    "error": f"Rate limit exceeded: {str(e)}",
                    "error_type": "rate_limit",
                    "status_code": getattr(e, "status_code", None),
                    "latency": elapsed
                }
                
            except ModelError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] Model error after {elapsed:.3f}s: {str(e)}")
                return {
                    "success": False,
                    "error": f"Model error: {str(e)}",
                    "error_type": "model",
                    "status_code": getattr(e, "status_code", None),
                    "latency": elapsed
                }
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] Unexpected error after {elapsed:.3f}s: {str(e)}")
                return {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "error_type": "unknown",
                    "latency": elapsed
                }
        
        return wrapper
    
    return decorator

def classify_http_error(status_code: int, response_data: Dict[str, Any]) -> APIError:
    """
    Classify an HTTP error based on status code and response data.
    
    Args:
        status_code: HTTP status code
        response_data: Response data from the API
        
    Returns:
        An appropriate APIError subclass instance
    """
    error_message = response_data.get("error", {}).get("message", "Unknown error")
    
    if status_code == 401 or status_code == 403:
        return AuthenticationError(
            message=f"Authentication failed: {error_message}",
            status_code=status_code,
            response=response_data
        )
    elif status_code == 429:
        return RateLimitError(
            message=f"Rate limit exceeded: {error_message}",
            status_code=status_code,
            response=response_data
        )
    elif status_code >= 500:
        return NetworkError(
            message=f"Server error ({status_code}): {error_message}",
            status_code=status_code,
            response=response_data
        )
    elif status_code == 400:
        # Try to determine if it's a model error
        if any(keyword in error_message.lower() for keyword in 
               ["context", "token", "prompt", "input", "parameter"]):
            return ModelError(
                message=f"Model error: {error_message}",
                status_code=status_code,
                response=response_data
            )
        else:
            return UnknownAPIError(
                message=f"Bad request: {error_message}",
                status_code=status_code,
                response=response_data
            )
    else:
        return UnknownAPIError(
            message=f"Unexpected error ({status_code}): {error_message}",
            status_code=status_code,
            response=response_data
        ) 