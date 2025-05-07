"""
Simplified Rule-Based Router for OpenRouter API

This module provides a rule-based router that selects the most appropriate model
based on prompt type (code, summary, question) and prompt length.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

# Simple logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rule_based_router")

class RuleBasedRouter:
    """
    Simplified rule-based router that selects models based on prompt type and length
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the router
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.models = config.get("models", {})
        self.default_model = config.get("default_model", "anthropic/claude-3-haiku")
        
        # Routing strategy - default to 'balanced'
        self.routing_strategy = config.get("optimization_target", "balanced")
        
        # Store the last model selection explanation
        self.model_selection_explanation = "No model selected yet"
        
        logger.info(f"Initialized simplified rule-based router with default model: {self.default_model}")
    
    def set_routing_strategy(self, strategy: str) -> None:
        """
        Set the routing strategy
        
        Args:
            strategy: Strategy to use ("balanced", "cost", "speed", "quality")
        """
        self.routing_strategy = strategy
        logger.info(f"Routing strategy set to: {strategy}")
    
    def classify_prompt(self, prompt: str) -> str:
        """
        Simplified function to classify the prompt type
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            Prompt classification ("code", "summary", "question")
        """
        # Check for code indicators
        if re.search(r'```|function|class|def|import|var|const|let', prompt, re.IGNORECASE):
            return "code"
        
        # Check for summary indicators
        elif re.search(r'summarize|summary|recap|tldr', prompt, re.IGNORECASE):
            return "summary"
        
        # Default to question
        else:
            return "question"
    
    def select_model(self, prompt: str) -> str:
        """
        Select an appropriate model based on the prompt
        
        Args:
            prompt: The user's prompt
            
        Returns:
            Selected model ID
        """
        # Get prompt type
        prompt_type = self.classify_prompt(prompt)
        
        # Simplified logic: use different models based on prompt type
        if prompt_type == "code":
            model_id = "openai/gpt-4o"
            reason = "Selected for coding tasks"
        elif prompt_type == "summary":
            model_id = "anthropic/claude-3-opus-20240229" 
            reason = "Selected for summarization tasks"
        else:  # question
            model_id = "anthropic/claude-3-haiku"
            reason = "Selected for general questions"
        
        # For cost strategy, always use the cheapest model
        if self.routing_strategy == "cost":
            model_id = "mistralai/mixtral-8x7b-instruct"
            reason = "Selected as the most cost-effective option"
        
        # For quality strategy, always use the most capable model
        elif self.routing_strategy == "quality":
            model_id = "anthropic/claude-3-opus-20240229"
            reason = "Selected as the highest quality option"
        
        # Fallback to default if model not in config
        if model_id not in self.models:
            model_id = self.default_model
            reason = "Fallback to default model"
        
        # Store explanation for the selection
        self.model_selection_explanation = (
            f"Model: {model_id}\n"
            f"Prompt type: {prompt_type}\n"
            f"Strategy: {self.routing_strategy}\n"
            f"Reason: {reason}"
        )
        
        return model_id
    
    def get_routing_explanation(self) -> Dict[str, Any]:
        """
        Get explanation for the routing decision
        
        Returns:
            Dictionary with explanation
        """
        return {
            "explanation": self.model_selection_explanation
        }
    
    def send_prompt(self, messages: List[Dict[str, str]], 
                   model_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Send the prompt to the selected model
        
        Args:
            messages: List of message dictionaries
            model_id: Optional model ID override
        
        Returns:
            Tuple of (response_text, metrics)
        """
        # Get the user's prompt from the messages
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            raise ValueError("No user message found in the provided messages")
        
        prompt = user_messages[-1]["content"]
        
        # Select model if not specified
        if not model_id:
            model_id = self.select_model(prompt)
        
        # Get model info
        model_info = self.models.get(model_id, {})
        temperature = model_info.get("temperature", 0.7)
        max_tokens = model_info.get("max_tokens", 1000)
        
        # Send to OpenRouter
        from ..utils.openrouter import send_request
        response_text, usage_stats, latency = send_request(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create metrics dictionary
        metrics = {
            "model": model_id,
            "prompt_type": self.classify_prompt(prompt),
            "token_count": usage_stats.get("total_tokens", 0),
            "prompt_tokens": usage_stats.get("prompt_tokens", 0),
            "completion_tokens": usage_stats.get("completion_tokens", 0),
            "latency": latency,
            "cost": (usage_stats.get("total_tokens", 0) * model_info.get("cost_per_1k_tokens", 0)) / 1000
        }
        
        return response_text, metrics 