#!/usr/bin/env python3
"""
Test the enhanced classification system with GPT-4 fallback.
This script tests how the classification system performs on ambiguous prompts.
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any

# Set up path to import from app directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our classifier
from app.utils.rule_based_router import PromptClassifier, EnhancedRuleBasedRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_classification")

# Sample ambiguous prompts that might benefit from GPT-4 classification
TEST_PROMPTS = [
    "I'm looking at this data and not sure what to make of it. Can you help me understand?",
    "I need to present this to the team tomorrow. Any advice?",
    "Can you take a look at these results and tell me what you think?",
    "I'm designing a system but I'm not sure about the approach.",
    "The client asked for something different than what we discussed. Need to pivot.",
    "This project has multiple stages and I'm trying to coordinate everything.",
    "Help me brainstorm some ideas for my upcoming presentation.",
    "I've collected some information but I'm not sure if I'm on the right track.",
    "Need to find a way to explain this complex topic to beginners.",
    "Trying to decide between several options for my project."
]

# Test prompts with clearer classifications for comparison
CONTROL_PROMPTS = [
    "Write a Python function to sort a list of tuples by the second element.",  # code
    "Summarize the key points from the following article: The effects of climate change...",  # summary
    "Analyze the pros and cons of remote work versus office-based employment.",  # analysis
    "Write a short story about a detective who can talk to animals.",  # creative
    "What is the capital of France?",  # question
]

async def test_classifier():
    """Test the classifier with and without GPT-4 fallback"""
    # Create classifier instances
    rule_only_classifier = PromptClassifier()
    rule_only_classifier.use_gpt4_fallback = False
    
    gpt4_fallback_classifier = PromptClassifier()
    gpt4_fallback_classifier.use_gpt4_fallback = True
    
    # Test control prompts first (should be classified correctly by rules)
    print("\n=== Testing Control Prompts (Clear Classification) ===")
    print("-" * 80)
    
    for prompt in CONTROL_PROMPTS:
        # Test rule-based classification
        prompt_type, details = rule_only_classifier.classify_sync(prompt)
        
        print(f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}")
        print(f"Rule-Based Classification: {prompt_type}")
        print(f"Method: {details.get('method', 'unknown')}")
        print(f"Details: {json.dumps(details, indent=2)}")
        print("-" * 80)
    
    # Test ambiguous prompts
    print("\n=== Testing Ambiguous Prompts ===")
    print("-" * 80)
    
    for prompt in TEST_PROMPTS:
        print(f"Prompt: {prompt}")
        
        # First test with rule-based only
        rule_result, rule_details = rule_only_classifier.classify_sync(prompt)
        
        print(f"Rule-Based Classification: {rule_result}")
        print(f"Method: {rule_details.get('method', 'unknown')}")
        
        # Then test with GPT-4 fallback
        try:
            gpt4_result, gpt4_details = await gpt4_fallback_classifier.classify(prompt)
            
            print(f"GPT-4 Fallback Classification: {gpt4_result}")
            print(f"Method: {gpt4_details.get('method', 'unknown')}")
            print(f"Confidence: {gpt4_details.get('confidence', 'unknown')}")
            
            # Check if classifications differ
            if rule_result != gpt4_result:
                print(f"DIFFERENT CLASSIFICATIONS: Rule={rule_result}, GPT-4={gpt4_result}")
        except Exception as e:
            print(f"Error with GPT-4 fallback: {str(e)}")
        
        print("-" * 80)

async def test_router():
    """Test the router with different prompts"""
    router = EnhancedRuleBasedRouter()
    
    print("\n=== Testing Router Model Selection with Ambiguous Prompts ===")
    print("-" * 80)
    
    for prompt in TEST_PROMPTS:
        print(f"Prompt: {prompt}")
        
        try:
            # Test with async version (GPT-4 fallback)
            model_id, reason, details = await router.select_model_async(prompt, "balanced")
            
            print(f"Selected Model (with GPT-4 fallback): {model_id}")
            print(f"Reason: {reason}")
            print(f"Classification: {details.get('prompt_type', 'unknown')}")
            print(f"Classification Method: {details.get('classification_details', {}).get('method', 'unknown')}")
            
            # Test with sync version (rule-based only)
            sync_model_id, sync_reason, sync_details = router.select_model(prompt, "balanced")
            
            print(f"Selected Model (rule-based only): {sync_model_id}")
            print(f"Reason: {sync_reason}")
            print(f"Classification: {sync_details.get('prompt_type', 'unknown')}")
            
            # Check if model selections differ
            if model_id != sync_model_id:
                print(f"DIFFERENT MODEL SELECTIONS: Async={model_id}, Sync={sync_model_id}")
        except Exception as e:
            print(f"Error with router: {str(e)}")
        
        print("-" * 80)

async def main():
    """Main entry point for the test script"""
    # Check if OpenRouter API key is set
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("WARNING: OPENROUTER_API_KEY not set in environment variables.")
        print("GPT-4 fallback will not work without an API key.")
        print("Please set the API key and try again.\n")
    
    # Test the classifier
    await test_classifier()
    
    # Test the router
    await test_router()

if __name__ == "__main__":
    asyncio.run(main()) 