"""
Model router module to select the best model based on the task type and routing strategy.
"""
from typing import Dict, List, Any, Tuple, Literal
from .task_detector import TaskType

RoutingStrategy = Literal["fastest", "cheapest", "most_capable", "balanced"]

def route_to_best_model(
    task_type: TaskType,
    routing_strategy: RoutingStrategy,
    config: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Route to the best model based on the task type and routing strategy.
    
    Args:
        task_type: The type of task (text-generation, code-generation, summarization)
        routing_strategy: The routing strategy to use
        config: The model configuration data
        
    Returns:
        Tuple of (model_id, explanation)
    """
    # Get available models for the task type
    available_models = config["models"].get(task_type, [])
    
    if not available_models:
        # Fallback to text-generation models if no specific models for the task
        available_models = config["models"].get("text-generation", [])
        if not available_models:
            return "anthropic/claude-3-haiku-20240307", "No models found for the task type, falling back to default model."
    
    # Select model based on routing strategy
    if routing_strategy == "fastest":
        # Sort by latency (ascending)
        sorted_models = sorted(available_models, key=lambda m: m.get("latency", float("inf")))
        selected_model = sorted_models[0]
        explanation = f"Selected the fastest model for {task_type} with a latency of {selected_model['latency']} seconds."
    
    elif routing_strategy == "cheapest":
        # Sort by cost (ascending)
        sorted_models = sorted(available_models, key=lambda m: m.get("cost_per_1k_tokens", float("inf")))
        selected_model = sorted_models[0]
        explanation = f"Selected the cheapest model for {task_type} at ${selected_model['cost_per_1k_tokens']} per 1K tokens."
    
    elif routing_strategy == "most_capable":
        # For simplicity, we'll assume the most expensive model is the most capable
        sorted_models = sorted(available_models, key=lambda m: m.get("cost_per_1k_tokens", 0), reverse=True)
        selected_model = sorted_models[0]
        explanation = (f"Selected the most capable model for {task_type}. "
                      f"This model excels at: {', '.join(selected_model.get('strengths', []))}")
    
    else:  # balanced
        # Create a scoring function based on normalized latency and cost
        def score_model(model):
            # Lower is better for both metrics
            latency = model.get("latency", float("inf"))
            cost = model.get("cost_per_1k_tokens", float("inf"))
            
            # Normalize between 0 and 1 (if we have multiple models)
            max_latency = max(m.get("latency", 0) for m in available_models)
            min_latency = min(m.get("latency", float("inf")) for m in available_models)
            latency_range = max_latency - min_latency
            
            max_cost = max(m.get("cost_per_1k_tokens", 0) for m in available_models)
            min_cost = min(m.get("cost_per_1k_tokens", float("inf")) for m in available_models)
            cost_range = max_cost - min_cost
            
            norm_latency = (latency - min_latency) / latency_range if latency_range > 0 else 0
            norm_cost = (cost - min_cost) / cost_range if cost_range > 0 else 0
            
            # Lower score is better (weighted combination)
            return 0.5 * norm_latency + 0.5 * norm_cost
        
        sorted_models = sorted(available_models, key=score_model)
        selected_model = sorted_models[0]
        explanation = (f"Selected a balanced model for {task_type} considering both performance and cost. "
                      f"Model latency: {selected_model['latency']} seconds, "
                      f"Cost: ${selected_model['cost_per_1k_tokens']} per 1K tokens.")
    
    return selected_model["id"], explanation 

def get_default_model() -> Tuple[str, str]:
    """Get the default model and reason for using it."""
    return "anthropic/claude-3-haiku", "No models found for the task type, falling back to default model." 