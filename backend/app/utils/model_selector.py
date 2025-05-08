"""
Model Selector for AI Routing

This module provides functionality to analyze prompts and select the most
appropriate AI model based on task type and routing strategy.
"""

import re
import logging
import yaml
import os
from typing import Dict, Any, Tuple, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_selector")

# Load config file
def load_config():
    """Load configuration from config.yaml file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
        return {"models": {}, "default_model": "anthropic/claude-3-haiku"}

# Global config
CONFIG = load_config()

def analyze_prompt(prompt: str) -> Dict[str, Any]:
    """
    Analyze a prompt to determine its characteristics
    
    Args:
        prompt: The text prompt to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Determine prompt type using the improved classifier
    prompt_type = classify_prompt_type(prompt)
    
    # Estimate length
    token_count = estimate_tokens(prompt)
    length_category = categorize_length(token_count)
    
    # Extract any specific model requests
    requested_model = extract_requested_model(prompt)
    
    return {
        "prompt_type": prompt_type,
        "token_count": token_count,
        "length_category": length_category,
        "requested_model": requested_model
    }

def classify_prompt_type(prompt: str) -> str:
    """
    Classify the prompt into a task type category using patterns from config
    
    Args:
        prompt: The prompt text
        
    Returns:
        Task type classification: "code", "creative", "analysis", "quick_questions", or "general"
    """
    # Get patterns from config
    prompt_types = CONFIG.get("prompt_types", {})
    
    # Check prompt against each type's patterns
    for prompt_type, type_config in prompt_types.items():
        patterns = type_config.get("patterns", [])
        for pattern in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.info(f"Classified prompt as {prompt_type} based on pattern match: {pattern}")
                return prompt_type
    
    # If no pattern matched, use fallback classification with hardcoded patterns
    
    # Code patterns
    code_patterns = [
        r'```\w*',             # Code blocks
        r'function\s+\w+\s*\(', # Function declarations
        r'class\s+\w+',         # Class declarations
        r'def\s+\w+\s*\(',      # Python function definitions
        r'import\s+\w+',        # Import statements
        r'\bvar\b|\blet\b|\bconst\b', # JavaScript variables
        r'SELECT.*FROM',        # SQL queries
        r'<\w+>.*</\w+>'        # HTML/XML tags
    ]
    
    # Check for code patterns
    for pattern in code_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return "code"
    
    # Summary/analysis patterns
    if any(keyword in prompt.lower() for keyword in ["summarize", "summary", "analyze", "explain", "compare"]):
        return "analysis"
        
    # Creative patterns
    if any(keyword in prompt.lower() for keyword in ["write", "create", "design", "generate", "story", "poem"]):
        return "creative"
        
    # Check if it's a question
    if prompt.strip().endswith("?") or prompt.lower().startswith(("what", "how", "why", "where", "when", "who", "can", "could")):
        return "quick_questions"
    
    # Default to general if no match
    return "general"

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Simple approximation: ~4 characters per token for English text
    return len(text) // 4

def categorize_length(token_count: int) -> str:
    """
    Categorize the prompt length
    
    Args:
        token_count: Number of tokens
        
    Returns:
        Length category: "short", "medium", or "long"
    """
    if token_count < 500:
        return "short"
    elif token_count < 2000:
        return "medium"
    else:
        return "long"

def extract_requested_model(prompt: str) -> Optional[str]:
    """
    Check if the prompt explicitly requests a specific model
    
    Args:
        prompt: The prompt text
        
    Returns:
        Model ID if explicitly requested, None otherwise
    """
    # Check for patterns like "use GPT-4" or "using Claude"
    model_mentions = {
        r'\bgpt-?4\b': "openai/gpt-4",
        r'\bgpt-?4o\b': "openai/gpt-4",
        r'\bclaude-?3\b': "anthropic/claude-3-opus",
        r'\bclaude\b': "anthropic/claude-3-opus",
        r'\bmixtral\b': "mistralai/mixtral-8x7b-instruct",
        r'\bmistral\b': "mistralai/mistral-7b-instruct",
        r'\bllama\b': "meta-llama/llama-2-70b-chat"
    }
    
    for pattern, model_id in model_mentions.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            logger.info(f"User explicitly requested model: {model_id}")
            return model_id
    
    return None

def select_model(
    prompt: str,
    strategy: str = "balanced",
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Tuple[str, str]:
    """
    Select the most appropriate model based on the prompt and strategy.
    
    Args:
        prompt: The user's prompt
        strategy: The routing strategy to use
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (model_id, reason)
    """
    # Start processing time
    logger.info(f"Selecting model for prompt with strategy: {strategy}")
    
    # Check if user explicitly requested a model
    requested_model = extract_requested_model(prompt)
    if requested_model:
        return requested_model, f"Selected {requested_model} as explicitly requested by user"
    
    # Analyze the prompt
    analysis = analyze_prompt(prompt)
    prompt_type = analysis["prompt_type"]
    length_category = analysis["length_category"]
    
    # Check for any strategy-specific default model in config
    strategy_config = CONFIG.get("routing_strategies", {}).get(strategy, {})
    strategy_default = strategy_config.get("default_model")
    
    # Try to use rule-based router from config
    rule_based = CONFIG.get("rule_based_router", {})
    
    # Map our prompt types to the rule_based_router categories
    rule_based_type = "question_models"  # Default
    if prompt_type == "code":
        rule_based_type = "code_models"
    elif prompt_type in ["analysis", "summary"]:
        rule_based_type = "summary_models"
    elif prompt_type in ["creative", "quick_questions"]:
        rule_based_type = "question_models"
    
    # Try to get the model from rule-based router based on prompt type and length
    rule_based_models = rule_based.get(rule_based_type, {})
    model_from_rules = rule_based_models.get(length_category)
    
    # Get preferred model from prompt type config
    prompt_type_config = CONFIG.get("prompt_types", {}).get(prompt_type, {})
    preferred_model = prompt_type_config.get("preferred_model")
    
    # Decision logic based on strategy
    if strategy == "cost":
        # Prioritize cost - use rule-based recommendation first, then strategy default
        model_id = model_from_rules or strategy_default or CONFIG.get("default_model", "anthropic/claude-3-haiku")
        reason = f"Selected {model_id} for cost optimization with {prompt_type} task ({length_category} length)"
        
    elif strategy == "speed":
        # Prioritize speed - smaller/faster models
        if prompt_type == "code" and length_category != "long":
            model_id = "openai/gpt-4"  # Good balance for coding
        elif length_category == "short":
            model_id = "anthropic/claude-3-haiku"  # Fastest for short tasks
        else:
            model_id = model_from_rules or "anthropic/claude-3-sonnet"  # Good balance
        reason = f"Selected {model_id} for speed optimization with {prompt_type} task ({length_category} length)"
        
    elif strategy == "quality":
        # Prioritize quality - use the most capable models
        if prompt_type == "code":
            model_id = "openai/gpt-4"  # Best for code
        else:
            model_id = "anthropic/claude-3-opus"  # Best for most other tasks
        reason = f"Selected {model_id} for quality optimization with {prompt_type} task"
        
    else:  # balanced strategy
        # For balanced, use rule-based router if available
        model_id = model_from_rules or preferred_model
        
        # If still not set, use hardcoded defaults based on prompt type
        if not model_id:
            if prompt_type == "code":
                model_id = "openai/gpt-4"  # Good for code
            elif prompt_type == "analysis":
                model_id = "anthropic/claude-3-opus"  # Good for analysis
            elif prompt_type == "creative":
                model_id = "anthropic/claude-3-sonnet"  # Good for creative
            elif prompt_type == "quick_questions":
                model_id = "anthropic/claude-3-haiku"  # Good for quick questions
            else:
                model_id = CONFIG.get("default_model", "anthropic/claude-3-haiku")
                
        reason = f"Selected {model_id} for balanced optimization with {prompt_type} task ({length_category} length)"
    
    # Ensure model exists in config, otherwise fall back to default
    default_model = CONFIG.get("default_model", "anthropic/claude-3-haiku")
    if model_id not in CONFIG.get("models", {}):
        model_id = default_model
        reason = f"Selected model not available, falling back to default: {default_model}"
    
    logger.info(f"Selected model: {model_id}, reason: {reason}")
    return model_id, reason 