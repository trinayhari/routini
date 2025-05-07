#!/usr/bin/env python
"""
Simple test script for the FastAPI backend

This script tests the generate and compare endpoints to ensure they're working correctly.
"""

import asyncio
import httpx
import json
import time
import argparse
from typing import Dict, Any, List

# Base URL for the API
DEFAULT_BASE_URL = "http://localhost:8000"

# Test models
TEST_MODELS = [
    "openai/gpt-4",
    "anthropic/claude-3-opus",
    "mistralai/mixtral-8x7b-instruct"
]

async def test_generate(client: httpx.AsyncClient, base_url: str) -> None:
    """Test the /generate endpoint with different prompts and strategies"""
    print("\n=== Testing /generate endpoint ===")
    
    # Test data
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "routing_strategy": "balanced",
            "expected_contains": "Paris"
        },
        {
            "prompt": "Write a simple function to calculate fibonacci numbers in Python",
            "routing_strategy": "quality",
            "expected_contains": "def fibonacci"
        },
        {
            "prompt": "Summarize the key benefits of cloud computing",
            "routing_strategy": "speed",
            "expected_contains": "cloud"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['prompt'][:30]}... with {test_case['routing_strategy']} strategy")
        
        try:
            start_time = time.time()
            response = await client.post(
                f"{base_url}/generate/",
                json={
                    "prompt": test_case["prompt"],
                    "routing_strategy": test_case["routing_strategy"],
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            )
            
            elapsed = time.time() - start_time
            print(f"  Response time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Success! Model used: {data['model_used']}")
                print(f"  Tokens: {data['token_metrics']['total']} | Cost: ${data['estimated_cost']:.6f}")
                print(f"  Explanation: {data['routing_explanation']}")
                
                # Simplified response text (first 100 chars)
                response_text = data['response']
                print(f"  Response: {response_text[:100]}...")
                
                # Verify expected content
                expected = test_case["expected_contains"].lower()
                if expected in response_text.lower():
                    print(f"  ✅ Contains expected string: '{expected}'")
                else:
                    print(f"  ❌ Expected string not found: '{expected}'")
            else:
                print(f"  ❌ Error {response.status_code}: {response.text}")
        
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")

async def test_compare(client: httpx.AsyncClient, base_url: str) -> None:
    """Test the /compare endpoint"""
    print("\n=== Testing /compare endpoint ===")
    
    try:
        print("\nTesting: Compare models for 'What is the capital of France?'")
        
        start_time = time.time()
        response = await client.post(
            f"{base_url}/compare/",
            json={
                "prompt": "What is the capital of France?",
                "models": TEST_MODELS,
                "max_tokens": 256,
                "temperature": 0.7
            }
        )
        
        elapsed = time.time() - start_time
        print(f"  Response time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Success! Compared {len(data['results'])} models:")
            
            for result in data['results']:
                model = result['model_id']
                latency = result['latency_seconds']
                cost = result['estimated_cost']
                tokens = result['token_count']['total']
                
                print(f"  - {model}:")
                print(f"    Latency: {latency:.2f}s | Cost: ${cost:.6f} | Tokens: {tokens}")
                print(f"    Response: {result['response'][:100]}...")
                
                # Check if "Paris" is in the response
                if "paris" in result['response'].lower():
                    print(f"    ✅ Contains 'Paris'")
                else:
                    print(f"    ❌ 'Paris' not found")
        else:
            print(f"  ❌ Error {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")

async def run_tests(base_url: str) -> None:
    """Run all tests"""
    print(f"Testing FastAPI backend at {base_url}")
    
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Test health endpoint
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            print("Is the backend server running?")
            return
        
        # Run all tests
        await test_generate(client, base_url)
        await test_compare(client, base_url)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test the FastAPI backend")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL for the API")
    args = parser.parse_args()
    
    asyncio.run(run_tests(args.url))

if __name__ == "__main__":
    main() 