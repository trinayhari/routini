"""
Test script to debug different components of the application.
"""
import asyncio
import yaml
from app.models.task_detector import detect_task_type
from app.models.model_router import route_to_best_model
from app.utils.config import get_config

async def test_task_detection():
    """Test task detection functionality"""
    prompts = [
        "Write a short poem about AI",
        "Create a Python function that calculates Fibonacci numbers",
        "Summarize the key points of quantum computing"
    ]
    
    for prompt in prompts:
        task_type = detect_task_type(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Detected task type: {task_type}")
        print()

async def test_model_routing():
    """Test model routing functionality"""
    config = get_config()
    
    tasks = ["text-generation", "code-generation", "summarization"]
    strategies = ["fastest", "cheapest", "most_capable", "balanced"]
    
    for task in tasks:
        print(f"\n=== Testing routing for {task} ===")
        for strategy in strategies:
            model_id, explanation = route_to_best_model(
                task_type=task,
                routing_strategy=strategy,
                config=config
            )
            print(f"Strategy: {strategy}")
            print(f"Selected model: {model_id}")
            print(f"Explanation: {explanation}")
            print()

async def main():
    print("=== Testing Task Detection ===")
    await test_task_detection()
    
    print("\n=== Testing Model Routing ===")
    await test_model_routing()

if __name__ == "__main__":
    asyncio.run(main()) 