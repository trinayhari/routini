"""
Advanced Router for OpenRouter API

This module provides an enhanced model router that uses performance metrics
to dynamically select the most appropriate model for a given prompt.
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd

from src.api.openrouter_client_enhanced import send_prompt_to_openrouter

# Set up logging
logging.basicConfig(
    filename=os.path.join("logs", "advanced_router.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

class AdvancedRouter:
    """
    Advanced router that uses performance metrics and content analysis
    to select the most appropriate model for each prompt.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_file: str = "model_metrics.csv"):
        """
        Initialize the router
        
        Args:
            config: Configuration dictionary with model settings
            metrics_file: CSV file to store/load model performance metrics
        """
        self.config = config
        self.models = config.get("models", {})
        self.prompt_types = config.get("prompt_types", {})
        self.default_model = config.get("default_model")
        self.metrics_file = metrics_file
        
        # Optimization settings
        self.optimization_target = config.get("optimization_target", "balanced")  # Options: speed, cost, quality, balanced
        
        # Initialize or load metrics
        self.metrics = self._load_metrics()
        
        # Initialize router parameters
        self._init_router_params()
    
    def _init_router_params(self):
        """Initialize router weights and parameters based on optimization target"""
        # Weights for different factors in model selection
        if self.optimization_target == "speed":
            self.weights = {
                "latency": 0.7,
                "token_efficiency": 0.2,
                "cost": 0.1,
                "pattern_match": 0.5
            }
        elif self.optimization_target == "cost":
            self.weights = {
                "latency": 0.1,
                "token_efficiency": 0.3,
                "cost": 0.6,
                "pattern_match": 0.5
            }
        elif self.optimization_target == "quality":
            self.weights = {
                "latency": 0.1,
                "token_efficiency": 0.2,
                "cost": 0.1,
                "pattern_match": 0.8
            }
        else:  # balanced
            self.weights = {
                "latency": 0.3,
                "token_efficiency": 0.3,
                "cost": 0.3,
                "pattern_match": 0.6
            }
    
    def _load_metrics(self) -> pd.DataFrame:
        """Load metrics from CSV file or initialize a new DataFrame"""
        try:
            if os.path.exists(self.metrics_file):
                metrics = pd.read_csv(self.metrics_file)
                logging.info(f"Loaded metrics from {self.metrics_file}: {len(metrics)} records")
                return metrics
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")
        
        # Initialize empty metrics DataFrame
        metrics = pd.DataFrame(columns=[
            "model", "timestamp", "prompt_type", "prompt_tokens", "completion_tokens",
            "total_tokens", "latency", "cost", "success"
        ])
        return metrics
    
    def _save_metrics(self):
        """Save current metrics to CSV file"""
        try:
            self.metrics.to_csv(self.metrics_file, index=False)
            logging.info(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
    
    def _update_metrics(self, model: str, prompt_type: str, 
                      usage_stats: Dict[str, Any], success: bool = True):
        """Update metrics with new data point"""
        new_metric = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "prompt_type": prompt_type,
            "prompt_tokens": usage_stats.get("prompt_tokens", 0),
            "completion_tokens": usage_stats.get("completion_tokens", 0),
            "total_tokens": usage_stats.get("total_tokens", 0),
            "latency": usage_stats.get("latency", 0),
            "cost": (usage_stats.get("total_tokens", 0) * 
                    self.models.get(model, {}).get("cost_per_1k_tokens", 0) / 1000),
            "success": success
        }
        
        # Append to metrics DataFrame
        self.metrics = pd.concat([self.metrics, pd.DataFrame([new_metric])], ignore_index=True)
        
        # Save updated metrics
        self._save_metrics()
    
    def identify_prompt_type(self, prompt: str) -> str:
        """
        Identify the type of prompt based on configured patterns
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            The identified prompt type or "general" if no match
        """
        for prompt_type, info in self.prompt_types.items():
            patterns = info.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    logging.info(f"Prompt matched pattern for type: {prompt_type}")
                    return prompt_type
        
        logging.info("No pattern matched, using general prompt type")
        return "general"
    
    def get_model_scores(self, prompt_type: str, prompt_tokens: int) -> Dict[str, float]:
        """
        Calculate a score for each model based on metrics and prompt type
        
        Args:
            prompt_type: The identified prompt type
            prompt_tokens: Estimated number of tokens in the prompt
            
        Returns:
            Dictionary of {model_id: score} where higher scores are better
        """
        scores = {}
        recent_metrics = self.metrics.tail(100)  # Use only recent metrics
        
        for model_id, model_info in self.models.items():
            # Start with base score
            score = 0.0
            
            # 1. Pattern match score
            preferred_model = self.prompt_types.get(prompt_type, {}).get("preferred_model")
            if model_id == preferred_model:
                score += 10.0 * self.weights["pattern_match"]
            
            # Filter metrics for this model
            model_metrics = recent_metrics[recent_metrics["model"] == model_id]
            
            if len(model_metrics) > 0:
                # 2. Latency score (lower is better)
                avg_latency = model_metrics["latency"].mean()
                if avg_latency > 0:
                    latency_score = 5.0 / avg_latency  # Normalize
                    score += latency_score * self.weights["latency"]
                
                # 3. Token efficiency score
                if "prompt_type" in model_metrics.columns:
                    type_metrics = model_metrics[model_metrics["prompt_type"] == prompt_type]
                    if len(type_metrics) > 0:
                        avg_tokens = type_metrics["total_tokens"].mean()
                        if avg_tokens > 0:
                            token_score = 1000.0 / avg_tokens  # Normalize
                            score += token_score * self.weights["token_efficiency"]
            
            # 4. Cost score (lower is better)
            cost_per_1k = model_info.get("cost_per_1k_tokens", 0.01)
            expected_total_tokens = prompt_tokens * 1.5  # Rough estimate
            expected_cost = expected_total_tokens * cost_per_1k / 1000
            
            if expected_cost > 0:
                cost_score = 0.05 / expected_cost  # Normalize
                score += cost_score * self.weights["cost"]
            
            # Store final score
            scores[model_id] = score
        
        logging.info(f"Model scores for {prompt_type}: {scores}")
        return scores
    
    def select_model(self, prompt: str, estimated_tokens: Optional[int] = None) -> str:
        """
        Select the best model for a prompt based on its content and metrics
        
        Args:
            prompt: The user's prompt
            estimated_tokens: Optional estimate of token count in prompt
            
        Returns:
            model_id: The ID of the selected model
        """
        # Identify prompt type
        prompt_type = self.identify_prompt_type(prompt)
        
        # Estimate tokens if not provided
        if estimated_tokens is None:
            # Very rough estimate: ~1 token per 4 chars for English
            estimated_tokens = len(prompt) // 4
        
        # Get scores for each model
        scores = self.get_model_scores(prompt_type, estimated_tokens)
        
        # Select model with highest score
        if scores:
            selected_model = max(scores, key=scores.get)
            score = scores[selected_model]
            logging.info(f"Selected model {selected_model} with score {score:.2f}")
        else:
            selected_model = self.default_model
            logging.info(f"No scores available, using default model: {selected_model}")
        
        return selected_model
    
    def send_prompt(self, messages: List[Dict[str, str]], 
                   model_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Send the prompt to the selected model and update metrics
        
        Args:
            messages: List of message dictionaries
            model_id: Optional model ID override
        
        Returns:
            Tuple of (response_text, usage_stats)
        """
        # Get the user's prompt from the messages
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            raise ValueError("No user message found in the provided messages")
        
        prompt = user_messages[-1]["content"]
        
        # Select model if not specified
        if not model_id:
            model_id = self.select_model(prompt)
        
        # Get model-specific parameters
        model_info = self.models.get(model_id, {})
        temperature = model_info.get("temperature", 0.7)
        max_tokens = model_info.get("max_tokens", 1000)
        
        # Identify prompt type for metrics
        prompt_type = self.identify_prompt_type(prompt)
        
        try:
            # Send to OpenRouter
            response_text, usage_stats, _ = send_prompt_to_openrouter(
                messages=messages,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Update metrics
            self._update_metrics(model_id, prompt_type, usage_stats, success=True)
            
            return response_text, usage_stats
            
        except Exception as e:
            logging.error(f"Error sending prompt to {model_id}: {e}")
            
            # Update metrics with failure
            self._update_metrics(
                model_id, 
                prompt_type, 
                {"latency": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 
                success=False
            )
            
            # Try fallback model if different from current model
            if model_id != self.default_model:
                logging.info(f"Trying fallback model: {self.default_model}")
                return self.send_prompt(messages, self.default_model)
            
            # Propagate the error if no fallback available
            raise 