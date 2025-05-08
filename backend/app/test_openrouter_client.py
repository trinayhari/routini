#!/usr/bin/env python3
"""
Test script for the OpenRouter client.
This script demonstrates how to use the OpenRouter client for various tasks.
"""

import os
import sys
import asyncio
import argparse
from typing import Dict, Any, List

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenRouter client
from app.utils.openrouter_client import ask, chat, compare

async def test_simple_question(model: str = "anthropic/claude-3-haiku"):
    """Test sending a simple question to the API"""
    print(f"\n=== Testing Simple Question (Model: {model}) ===")
    
    prompt = "What are the three main principles of effective communication?"
    print(f"Question: {prompt}")
    print("Waiting for response...")
    
    response = await ask(prompt, model=model, max_tokens=300)
    
    print("\nResponse:")
    print("="*70)
    
    if response["success"]:
        print(response["text"])
        print("="*70)
        print(f"Model: {response['model']}")
        print(f"Tokens: {response['tokens']} (prompt: {response['prompt_tokens']}, completion: {response['completion_tokens']})")
        print(f"Cost: ${response['cost']:.6f} | Latency: {response['latency']:.2f}s")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    return response

async def test_chat_conversation(model: str = "anthropic/claude-3-opus"):
    """Test a multi-turn chat conversation"""
    print(f"\n=== Testing Chat Conversation (Model: {model}) ===")
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that specializes in Python programming."},
        {"role": "user", "content": "Write a Python function to check if a string is a palindrome."}
    ]
    
    print("Starting conversation with system prompt and user question about palindromes...")
    print("Waiting for response...")
    
    response = await chat(messages, model=model, max_tokens=500)
    
    if response["success"]:
        print("\nAssistant:")
        print("-"*70)
        print(response["text"])
        print("-"*70)
        print(f"Tokens: {response['tokens']} | Cost: ${response['cost']:.6f}")
        
        # Continue the conversation
        messages.append({"role": "assistant", "content": response["text"]})
        messages.append({"role": "user", "content": "Can you explain the time complexity of this function?"})
        
        print("\nUser: Can you explain the time complexity of this function?")
        print("Waiting for response...")
        
        response2 = await chat(messages, model=model, max_tokens=300)
        
        if response2["success"]:
            print("\nAssistant:")
            print("-"*70)
            print(response2["text"])
            print("-"*70)
            print(f"Tokens: {response2['tokens']} | Cost: ${response2['cost']:.6f}")
        else:
            print(f"Error: {response2.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    return response

async def test_model_comparison():
    """Test comparing multiple models on the same prompt"""
    print("\n=== Testing Model Comparison ===")
    
    prompt = "Explain how blockchain technology works in simple terms."
    models = [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet",
        "openai/gpt-4"
    ]
    
    print(f"Prompt: {prompt}")
    print(f"Comparing {len(models)} models: {', '.join(models)}")
    print("Waiting for responses...")
    
    comparison = await compare(prompt, models, max_tokens=300)
    
    print("\nResults:")
    
    for result in comparison["results"]:
        model = result["model"]
        print("\n" + "="*70)
        print(f"Model: {model}")
        print("="*70)
        
        if result["success"]:
            print(result["text"])
            print("-"*70)
            print(f"Tokens: {result['tokens']} | Cost: ${result['cost']:.6f} | Time: {result['latency']:.2f}s")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nTotal comparison time: {comparison['total_latency']:.2f}s")
    
    return comparison

async def main():
    """Run tests based on command line arguments"""
    parser = argparse.ArgumentParser(description="Test OpenRouter client")
    parser.add_argument("--test", choices=["question", "chat", "compare", "all"],
                        default="all", help="Test to run")
    parser.add_argument("--model", type=str, default="anthropic/claude-3-haiku",
                        help="Model to use for tests")
    args = parser.parse_args()
    
    print("OpenRouter Client Tests")
    print("======================")
    
    if args.test == "question" or args.test == "all":
        await test_simple_question(args.model)
    
    if args.test == "chat" or args.test == "all":
        await test_chat_conversation(args.model)
    
    if args.test == "compare" or args.test == "all":
        await test_model_comparison()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 