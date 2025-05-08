#!/usr/bin/env python3
"""
Examples of using the OpenRouter API client.

This module demonstrates different ways to use the OpenRouter API client
for sending prompts to various language models.
"""

import os
import sys
import asyncio
import time
from typing import List, Dict, Any

# Add the parent directory to sys.path to import from app
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# Import the OpenRouter client functions
from app.utils.openrouter import send_request

# -------------------------------------------------------------------------------
# Example 1: Simple query
# -------------------------------------------------------------------------------

async def example_simple_query():
    """
    Demonstrate a simple question-answering query.
    """
    print("\n===== Example 1: Simple Query =====")
    
    prompt = "What are the three main components of a modern LLM architecture?"
    
    print(f"Prompt: {prompt}")
    print("Waiting for response...")
    
    try:
        # Create a message in the format expected by the API
        messages = [{"role": "user", "content": prompt}]
        
        # Send the request directly using the async API
        response = await send_request(
            model="anthropic/claude-3-haiku",  # Using a faster model for quick responses
            messages=messages,
            max_tokens=300
        )
        
        # Check for errors
        if isinstance(response, dict) and "error" in response:
            print(f"Error: {response['message']}")
            return None
            
        # Unpack the successful response
        response_text, usage_stats, latency = response
        
        print(f"\nResponse (in {latency:.2f}s):")
        print("-" * 70)
        print(response_text)
        print("-" * 70)
        print(f"Tokens: {usage_stats['total_tokens']} (prompt: {usage_stats['prompt_tokens']}, completion: {usage_stats['completion_tokens']})")
        print(f"Cost: ${usage_stats['cost']:.6f}")
        
        return response_text
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# -------------------------------------------------------------------------------
# Example 2: Chat conversation
# -------------------------------------------------------------------------------

async def example_chat_conversation():
    """
    Demonstrate a multi-turn chat conversation using the async API directly.
    """
    print("\n===== Example 2: Chat Conversation =====")
    
    # Create a conversation history
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with expertise in computer science."},
        {"role": "user", "content": "Can you explain what a transformer architecture is?"},
    ]
    
    # First user question
    print("User: Can you explain what a transformer architecture is?")
    print("Waiting for response...")
    
    # Get assistant response
    response = await send_request(
        model="openai/gpt-4",
        messages=messages,
        max_tokens=400
    )
    
    # Check for errors
    if isinstance(response, dict) and "error" in response:
        print(f"Error: {response['message']}")
        return
    
    # Unpack the response
    response_text, usage_stats, latency = response
    
    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": response_text})
    
    # Print the response
    print(f"\nAssistant ({latency:.2f}s):")
    print("-" * 70)
    print(response_text)
    print("-" * 70)
    print(f"Tokens: {usage_stats['total_tokens']} | Cost: ${usage_stats['cost']:.6f}")
    
    # Second user question - followup
    follow_up = "How does this differ from RNN architectures?"
    print(f"\nUser: {follow_up}")
    print("Waiting for response...")
    
    # Add user message to conversation
    messages.append({"role": "user", "content": follow_up})
    
    # Get assistant response for the follow-up
    response = await send_request(
        model="openai/gpt-4",
        messages=messages,
        max_tokens=400
    )
    
    # Check for errors
    if isinstance(response, dict) and "error" in response:
        print(f"Error: {response['message']}")
        return
    
    # Unpack the response
    response_text, usage_stats, latency = response
    
    # Print the response
    print(f"\nAssistant ({latency:.2f}s):")
    print("-" * 70)
    print(response_text)
    print("-" * 70)
    print(f"Tokens: {usage_stats['total_tokens']} | Cost: ${usage_stats['cost']:.6f}")
    
    return messages

# -------------------------------------------------------------------------------
# Example 3: Comparing models
# -------------------------------------------------------------------------------

async def example_compare_models():
    """
    Demonstrate comparing responses from different models for the same prompt.
    """
    print("\n===== Example 3: Model Comparison =====")
    
    prompt = "Explain the concept of quantum entanglement in simple terms."
    messages = [{"role": "user", "content": prompt}]
    
    models = [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet",
        "openai/gpt-4"
    ]
    
    results = []
    
    print(f"Prompt: {prompt}")
    print(f"Querying {len(models)} models in parallel...")
    
    # Query each model in parallel
    async def query_model(model):
        start_time = time.time()
        
        try:
            response = await send_request(
                model=model,
                messages=messages,
                max_tokens=300
            )
            
            # If we got an error, return it
            if isinstance(response, dict) and "error" in response:
                return {
                    "model": model,
                    "error": response["message"],
                    "latency": time.time() - start_time
                }
            
            # Otherwise unpack the successful response
            response_text, usage_stats, _ = response
            
            return {
                "model": model,
                "response": response_text,
                "tokens": usage_stats["total_tokens"],
                "cost": usage_stats["cost"],
                "latency": time.time() - start_time
            }
        except Exception as e:
            return {
                "model": model,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    # Create tasks for all models
    tasks = [query_model(model) for model in models]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Print results
    for result in results:
        model = result["model"]
        print("\n" + "-" * 70)
        print(f"Model: {model}")
        print("-" * 70)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(result["response"])
            print("-" * 70)
            print(f"Tokens: {result['tokens']} | Cost: ${result['cost']:.6f} | Time: {result['latency']:.2f}s")
    
    return results

# -------------------------------------------------------------------------------
# Main function that allows running examples individually or all together
# -------------------------------------------------------------------------------

async def run_example(example_number):
    """Run a specific example"""
    if example_number == 1:
        await example_simple_query()
    elif example_number == 2:
        await example_chat_conversation()
    elif example_number == 3:
        await example_compare_models()
    else:
        print(f"Example {example_number} not found.")

async def main():
    """Run examples based on command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenRouter API Usage Examples")
    parser.add_argument("example", type=int, nargs="?", choices=[1, 2, 3], 
                        help="Example number to run (1, 2, or 3)")
    args = parser.parse_args()
    
    print("OpenRouter API Usage Examples")
    print("============================")
    
    if args.example:
        # Run just the specified example
        await run_example(args.example)
    else:
        # Run all examples
        print("Running all examples...")
        
        # Example 1: Simple Query
        await example_simple_query()
        
        # Example 2: Chat Conversation
        await example_chat_conversation()
        
        # Example 3: Comparing Models
        await example_compare_models()
    
    print("\nExamples completed successfully.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 