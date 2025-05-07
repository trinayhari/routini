#!/usr/bin/env python3
"""
Model Benchmarking Tool for OpenRouter

This script benchmarks different LLM models from OpenRouter for:
- Response quality
- Token efficiency
- Latency
- Cost

Usage:
    python benchmark_models.py
"""

import os
import json
import time
import pandas as pd
import argparse
from typing import List, Dict, Any
from src.api.openrouter_client_enhanced import send_prompt_to_openrouter
from src.config.config_loader import load_config

def run_benchmark(
    prompt: str, 
    models: List[str], 
    system_message: str = "You are a helpful assistant.",
    runs_per_model: int = 3,
    output_file: str = "benchmark_results.csv"
) -> pd.DataFrame:
    """
    Benchmark multiple models with the same prompt
    
    Args:
        prompt: The prompt to test
        models: List of model IDs to benchmark
        system_message: System message to use for all prompts
        runs_per_model: Number of runs per model for averaging results
        output_file: File to save benchmark results
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    # Config for cost calculations
    try:
        config = load_config()
        model_configs = config.get("models", {})
    except Exception as e:
        print(f"Warning: Could not load config for cost calculations: {e}")
        model_configs = {}
    
    # Run benchmarks
    print(f"Benchmarking {len(models)} models, {runs_per_model} runs each...")
    
    for model_id in models:
        print(f"\nTesting model: {model_id}")
        model_results = []
        
        for run in range(1, runs_per_model + 1):
            try:
                print(f"  Run {run}/{runs_per_model}...", end="", flush=True)
                
                # Call the model
                start_time = time.time()
                response_text, usage_stats, api_latency = send_prompt_to_openrouter(
                    messages=messages,
                    model=model_id,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Calculate total latency including function overhead
                total_latency = time.time() - start_time
                
                # Get cost per 1k tokens from config
                model_info = model_configs.get(model_id, {})
                cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
                
                # Calculate cost
                total_tokens = usage_stats.get("total_tokens", 0)
                estimated_cost = (total_tokens * cost_per_1k) / 1000
                
                # First 25 chars of response for quick verification
                response_preview = response_text[:25].replace("\n", " ") + "..."
                
                # Store results
                result = {
                    "model": model_id,
                    "run": run,
                    "prompt_tokens": usage_stats.get("prompt_tokens", 0),
                    "completion_tokens": usage_stats.get("completion_tokens", 0),
                    "total_tokens": total_tokens,
                    "api_latency": api_latency,
                    "total_latency": total_latency,
                    "estimated_cost": estimated_cost,
                    "response_preview": response_preview,
                    "response_text": response_text
                }
                
                model_results.append(result)
                print(f" Done. Tokens: {total_tokens}, Latency: {api_latency:.2f}s")
                
                # Small delay between calls
                time.sleep(1)
                
            except Exception as e:
                print(f" Error: {e}")
                # Store error result
                model_results.append({
                    "model": model_id,
                    "run": run,
                    "error": str(e)
                })
        
        # Add all runs to results
        results.extend(model_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nBenchmark Summary:")
    summary = df.groupby("model").agg({
        "prompt_tokens": "mean",
        "completion_tokens": "mean",
        "total_tokens": "mean",
        "api_latency": "mean",
        "total_latency": "mean",
        "estimated_cost": "mean"
    }).round(3)
    
    print(summary)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter models")
    parser.add_argument("--prompt", type=str, default="Explain the difference between deep learning and machine learning in 3 paragraphs.",
                      help="The prompt to use for benchmarking")
    parser.add_argument("--models", type=str, nargs="+", 
                      default=["anthropic/claude-3-haiku", "anthropic/claude-3-sonnet", "openai/gpt-3.5-turbo"],
                      help="List of models to benchmark")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.",
                      help="System message to use")
    parser.add_argument("--runs", type=int, default=3,
                      help="Number of runs per model")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                      help="Output file for results")
    
    args = parser.parse_args()
    
    run_benchmark(
        prompt=args.prompt,
        models=args.models,
        system_message=args.system,
        runs_per_model=args.runs,
        output_file=args.output
    ) 