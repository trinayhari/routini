#!/usr/bin/env python3
"""
Diagnostic tool for troubleshooting empty responses from models.
Run this script when you encounter empty response issues.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional

# Set up path to import from app directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the diagnostic function
from app.utils.openrouter import diagnose_empty_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("empty_response_diagnosis.log")
    ]
)
logger = logging.getLogger("diagnosis")

# Safe test prompts
SAFE_TEST_PROMPTS = [
    "Hello, how are you today?",
    "What is 2+2?",
    "Tell me about the solar system.",
    "Write a short poem about cats."
]

# Models to test
DEFAULT_MODELS = [
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "openai/gpt-4"
]

async def diagnose_model(
    model: str,
    prompts: List[str],
    custom_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Run diagnostic tests on a specific model.
    
    Args:
        model: The model to test
        prompts: List of safe prompts to test
        custom_prompt: Optional custom prompt to test
        max_tokens: Maximum tokens to generate
        temperature: Temperature setting
        
    Returns:
        Dictionary with diagnostic results
    """
    logger.info(f"Testing model: {model}")
    
    results = {
        "model": model,
        "successful_tests": 0,
        "failed_tests": 0,
        "empty_responses": 0,
        "test_results": []
    }
    
    # Test all the safe prompts
    test_prompts = list(prompts)
    
    # Add the custom prompt if provided
    if custom_prompt:
        test_prompts.append(custom_prompt)
    
    for prompt in test_prompts:
        logger.info(f"Testing prompt: '{prompt[:30]}...'")
        
        try:
            # Run the diagnostic
            diagnostic = await diagnose_empty_response(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Record the result
            if diagnostic["success"]:
                results["successful_tests"] += 1
                
                # Check for empty responses
                if diagnostic["is_empty_response"] or diagnostic["contains_error_message"]:
                    results["empty_responses"] += 1
                    logger.warning(f"Empty response detected for prompt: '{prompt[:30]}...'")
                else:
                    logger.info(f"Successful response: {len(diagnostic['response_text'])} chars")
                
                # Add the test result
                results["test_results"].append({
                    "prompt": prompt,
                    "success": True,
                    "is_empty": diagnostic["is_empty_response"],
                    "contains_error_msg": diagnostic["contains_error_message"],
                    "response_length": len(diagnostic["response_text"]),
                    "tokens": diagnostic["usage_stats"]["total_tokens"],
                    "latency": diagnostic["latency"]
                })
            else:
                results["failed_tests"] += 1
                
                # Add the failed test result
                results["test_results"].append({
                    "prompt": prompt,
                    "success": False,
                    "error": diagnostic.get("error", "Unknown error"),
                    "status_code": diagnostic.get("status_code")
                })
                
                logger.error(f"Test failed: {diagnostic.get('error', 'Unknown error')}")
        
        except Exception as e:
            # Record the exception
            results["failed_tests"] += 1
            results["test_results"].append({
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
            
            logger.error(f"Exception during test: {str(e)}")
    
    # Summarize the results
    logger.info(f"Model {model} summary: {results['successful_tests']} successful, "
               f"{results['failed_tests']} failed, {results['empty_responses']} empty responses")
    
    return results

async def run_diagnostics(
    models: List[str] = DEFAULT_MODELS,
    custom_prompt: Optional[str] = None,
    output_file: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Run diagnostics on multiple models.
    
    Args:
        models: List of models to test
        custom_prompt: Optional custom prompt to test
        output_file: Optional file to write results to
        max_tokens: Maximum tokens to generate
        temperature: Temperature setting
        
    Returns:
        Dictionary with all diagnostic results
    """
    logger.info(f"Starting diagnostic run on {len(models)} models")
    
    if custom_prompt:
        logger.info(f"Using custom prompt: '{custom_prompt[:50]}...'")
    
    all_results = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "models_tested": len(models),
        "models": {}
    }
    
    # Test each model
    for model in models:
        model_results = await diagnose_model(
            model=model,
            prompts=SAFE_TEST_PROMPTS,
            custom_prompt=custom_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        all_results["models"][model] = model_results
    
    # Summarize the overall results
    success_count = sum(
        results["successful_tests"]
        for results in all_results["models"].values()
    )
    
    fail_count = sum(
        results["failed_tests"]
        for results in all_results["models"].values()
    )
    
    empty_count = sum(
        results["empty_responses"]
        for results in all_results["models"].values()
    )
    
    all_results["summary"] = {
        "total_tests": success_count + fail_count,
        "successful_tests": success_count,
        "failed_tests": fail_count,
        "empty_responses": empty_count
    }
    
    logger.info(f"Diagnostic complete. Overall: {success_count} successful, "
               f"{fail_count} failed, {empty_count} empty responses")
    
    # Write results to file if requested
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Results written to {output_file}")
        except Exception as e:
            logger.error(f"Error writing results to {output_file}: {str(e)}")
    
    return all_results

async def main():
    """
    Main entry point for the diagnostic script.
    """
    parser = argparse.ArgumentParser(description="Diagnose empty response issues with LLMs")
    
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="List of models to test")
    
    parser.add_argument("--prompt", type=str,
                        help="Custom prompt to test (in addition to safe prompts)")
    
    parser.add_argument("--output", type=str, default="diagnosis_results.json",
                        help="File to write results to (default: diagnosis_results.json)")
    
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature setting")
    
    args = parser.parse_args()
    
    # Enable debug mode with environment variable
    os.environ["USE_MOCK"] = "true"
    
    try:
        # Run the diagnostics
        await run_diagnostics(
            models=args.models,
            custom_prompt=args.prompt,
            output_file=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        logger.info("Diagnosis complete. Check the output file for details.")
    
    except KeyboardInterrupt:
        logger.info("Diagnostic interrupted by user.")
    
    except Exception as e:
        logger.error(f"Error running diagnostics: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 