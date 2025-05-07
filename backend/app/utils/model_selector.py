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
    # Determine prompt type
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
    Classify the prompt into a task type category
    
    Args:
        prompt: The prompt text
        
    Returns:
        Task type classification: "code", "summary", or "question"
    """
    # Check for code-related patterns
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
    
    # Check for summarization patterns
    summary_patterns = [
        r'\bsummarize\b', 
        r'\bsummary\b', 
        r'\btl;?dr\b',
        r'\bcondense\b',
        r'\bsynthesize\b'
    ]
    
    # Check for each pattern
    for pattern in code_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return "code"
            
    for pattern in summary_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return "summary"
    
    # Default to question if no clear pattern match
    return "question"

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
    # Load configuration
    CONFIG = load_config()
    
    # Get prompt type
    prompt_type = classify_prompt_type(prompt)
    
    # Get token count
    token_count = estimate_tokens(prompt)
    
    # Select model based on strategy
    if strategy == "cost":
        # For cost optimization, use the cheapest model that can handle the task
        if prompt_type == "code":
            model_id = "openai/gpt-4"  # GPT-4 is best for coding
        elif prompt_type == "summary":
            model_id = "anthropic/claude-3-opus"  # Claude 3 Opus for analysis
        elif prompt_type == "question":
            model_id = "anthropic/claude-3-opus"  # Claude 3 Opus for creative tasks
        else:
            model_id = "openai/gpt-4"  # Default to GPT-4
        reason = f"Selected {model_id} for cost optimization with {prompt_type} task"
        
    elif strategy == "speed":
        # For speed optimization, use the fastest model that can handle the task
        if prompt_type == "code":
            model_id = "openai/gpt-4"  # GPT-4 is best for coding
        elif prompt_type == "summary":
            model_id = "anthropic/claude-3-opus"  # Claude 3 Opus for analysis
        elif prompt_type == "question":
            model_id = "anthropic/claude-3-sonnet"  # Claude 3 Sonnet for creative tasks
        else:
            model_id = "openai/gpt-4"  # Default to GPT-4
        reason = f"Selected {model_id} for speed optimization with {prompt_type} task"
        
    else:  # balanced strategy
        # For balanced optimization, consider task type and complexity
        if prompt_type == "code":
            model_id = "openai/gpt-4"  # GPT-4 is best for coding
        elif prompt_type == "summary":
            model_id = "anthropic/claude-3-opus"  # Claude 3 Opus for analysis
        elif prompt_type == "question":
            model_id = "anthropic/claude-3-sonnet"  # Claude 3 Sonnet for creative tasks
        else:
            model_id = "openai/gpt-4"  # Default to GPT-4
        reason = f"Selected {model_id} for balanced optimization with {prompt_type} task"
    
    # Get default model from config
    default_model = CONFIG.get("default_model", "anthropic/claude-3-haiku")
    
    # If selected model is not available, fall back to default
    if model_id not in CONFIG.get("models", {}):
        model_id = default_model
        reason = f"Selected model not available, falling back to default: {default_model}"
    
    return model_id, reason 