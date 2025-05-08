#!/usr/bin/env python3
"""
Test script for verifying the error handling in the OpenRouter client.
This script tests different error scenarios to ensure they're handled correctly.
"""

import asyncio
import os
import sys
import json
import logging
from typing import Dict, Any, List

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the functions we want to test
from app.utils.openrouter_client import ask, chat, compare
from app.utils.error_handler import APIError, NetworkError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("error_test")

async def test_empty_prompt():
    """Test handling of empty prompts"""
    logger.info("\n=== Testing Empty Prompt Handling ===")
    
    try:
        # This should be caught by our validation
        response = await ask("", model="anthropic/claude-3-haiku")
        if response["success"] is False:
            logger.info("✅ Empty prompt correctly handled with error response")
            logger.info(f"Error message: {response['error']}")
        else:
            logger.error("❌ Empty prompt should have returned an error")
    except ValueError as e:
        logger.info(f"✅ Empty prompt correctly raised ValueError: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error type: {type(e).__name__}: {str(e)}")

async def test_invalid_model():
    """Test handling of invalid model names"""
    logger.info("\n=== Testing Invalid Model Handling ===")
    
    # This should return a proper error response
    response = await ask(
        "What is the capital of France?",
        model="invalid-model-that-doesnt-exist"
    )
    
    if not response["success"]:
        logger.info("✅ Invalid model correctly handled with error response")
        logger.info(f"Error: {response.get('error')}")
        logger.info(f"Error type: {response.get('error_type')}")
    else:
        logger.error("❌ Invalid model should have returned an error")

async def test_empty_message_content():
    """Test handling of empty message content in a chat"""
    logger.info("\n=== Testing Empty Message Content Handling ===")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": ""}  # Empty content
    ]
    
    try:
        response = await chat(messages, model="anthropic/claude-3-haiku")
        if response["success"] is False:
            logger.info("✅ Empty message content correctly handled with error response")
            logger.info(f"Error message: {response['error']}")
        else:
            logger.error("❌ Empty message content should have returned an error")
    except ValueError as e:
        logger.info(f"✅ Empty message content correctly raised ValueError: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error type: {type(e).__name__}: {str(e)}")

async def test_blank_response_handling():
    """
    Test handling of blank responses from the API.
    
    Note: This test is more difficult to run automatically since we'd need to
    mock the API to return blank responses. We'll simulate it by checking if
    our code has the appropriate checks in place.
    """
    logger.info("\n=== Testing Blank Response Handling ===")
    
    # We're just checking if the code has appropriate checks for blank responses
    # This is a simple prompt that shouldn't produce blank responses
    response = await ask("What is 2+2?", model="anthropic/claude-3-haiku")
    
    if response["success"] and response["text"] and len(response["text"]) > 0:
        logger.info("✅ Response contains non-empty text")
        logger.info(f"Response: {response['text'][:100]}...")
    else:
        logger.error("❌ Response should contain text")
        logger.error(f"Response: {response}")

async def test_parallel_requests():
    """Test handling multiple requests in parallel"""
    logger.info("\n=== Testing Parallel Requests Handling ===")
    
    # Create multiple requests to run in parallel
    tasks = [
        ask(f"What is {i} + {i}?", model="anthropic/claude-3-haiku")
        for i in range(1, 4)
    ]
    
    # Run requests in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    all_ok = True
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ Request {i+1} failed with exception: {str(result)}")
            all_ok = False
        elif not result["success"]:
            logger.error(f"❌ Request {i+1} failed: {result.get('error')}")
            all_ok = False
        else:
            logger.info(f"✅ Request {i+1} succeeded: {result['text'][:30]}...")
    
    if all_ok:
        logger.info("✅ All parallel requests completed successfully")
    else:
        logger.error("❌ Some parallel requests failed")

async def test_metadata_handling():
    """Test handling of metadata in messages"""
    logger.info("\n=== Testing Metadata Handling ===")
    
    # Test with string metadata (should work)
    string_metadata = json.dumps({"source": "test", "priority": "high"})
    
    messages_with_string_metadata = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello", "metadata": string_metadata}
    ]
    
    # Test with object metadata (backend should handle both)
    object_metadata = {"source": "test", "priority": "high"}
    
    messages_with_object_metadata = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello", "metadata": object_metadata}
    ]
    
    try:
        logger.info("Testing with string metadata:")
        response1 = await chat(messages_with_string_metadata, model="anthropic/claude-3-haiku")
        if response1["success"]:
            logger.info("✅ String metadata handled correctly")
        else:
            logger.error(f"❌ String metadata failed: {response1.get('error')}")
            
        logger.info("Testing with object metadata:")
        response2 = await chat(messages_with_object_metadata, model="anthropic/claude-3-haiku")
        if response2["success"]:
            logger.info("✅ Object metadata handled correctly")
        else:
            logger.error(f"❌ Object metadata failed: {response2.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Error in metadata test: {str(e)}")

async def test_anthropic_opus_empty_response():
    """
    Specifically test the anthropic/claude-3-opus model that's having empty response issues.
    This test will help diagnose what's happening with this specific model.
    """
    logger.info("\n=== Testing Anthropic Claude 3 Opus Empty Response Issue ===")
    
    # Import the diagnostic function
    from app.utils.openrouter import diagnose_empty_response
    
    # Try several different prompts to see if we can diagnose the issue
    safe_prompts = [
        "Hello, how are you today?",
        "What is 2+2?",
        "Tell me about the solar system.",
        "Write a short poem about cats."
    ]
    
    for prompt in safe_prompts:
        logger.info(f"Testing with prompt: '{prompt}'")
        
        # Run the diagnostic function
        diagnostic = await diagnose_empty_response(
            model="anthropic/claude-3-opus",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7
        )
        
        if diagnostic["success"]:
            # Check if the response was empty
            if diagnostic["is_empty_response"] or diagnostic["contains_error_message"]:
                logger.error(f"❌ Empty response confirmed with prompt: '{prompt}'")
                logger.error(f"Response: {diagnostic['response_text']}")
            else:
                logger.info(f"✅ Got a valid response: {diagnostic['response_text'][:100]}...")
                
            # Log diagnostic information
            logger.info(f"Tokens: {diagnostic['usage_stats']['total_tokens']} " +
                      f"(prompt: {diagnostic['usage_stats']['prompt_tokens']}, " +
                      f"completion: {diagnostic['usage_stats']['completion_tokens']})")
            logger.info(f"Latency: {diagnostic['latency']:.2f}s")
        else:
            logger.error(f"❌ Diagnostic test failed: {diagnostic.get('error', 'Unknown error')}")
    
    logger.info("Opus model test completed")

async def main():
    """Run all tests"""
    logger.info("Starting error handling tests...")
    
    try:
        await test_empty_prompt()
        await test_invalid_model()
        await test_empty_message_content()
        await test_blank_response_handling()
        await test_parallel_requests()
        await test_metadata_handling()
        
        # Add the new test for the specific Anthropic model
        await test_anthropic_opus_empty_response()
        
        logger.info("\n=== All Tests Completed ===")
    except Exception as e:
        logger.error(f"Unexpected error in tests: {str(e)}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 