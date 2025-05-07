"""
Simplified Mock Cost Tracker

A utility module for tracking and analyzing API usage costs.
This is a simplified version to avoid errors in the FastAPI backend.
"""

import os
import logging
from typing import Dict, Any, Optional

# Set up logging
os.makedirs("logs", exist_ok=True)
cost_logger = logging.getLogger("cost_tracker")
cost_logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join("logs", "cost_tracking.log"))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
cost_logger.addHandler(handler)

class CostTracker:
    """
    Track and analyze API usage costs across multiple models
    Simplified mock implementation
    """
    
    def __init__(self, config: Dict[str, Any], log_file: str = "api_costs.csv"):
        """
        Initialize the cost tracker
        
        Args:
            config: Configuration dictionary with model pricing
            log_file: CSV file to log costs to
        """
        self.config = config
        self.models = config.get("models", {})
        self.log_file = os.path.join("logs", log_file)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Basic initialization
        self.current_session_id = "mock_session"
        cost_logger.info(f"Mock cost tracker initialized with session ID: {self.current_session_id}")
    
    def log_api_call(self, model: str, usage_stats: Dict[str, Any], session_id: Optional[str] = None) -> float:
        """
        Log an API call and calculate its cost
        
        Args:
            model: The model used
            usage_stats: Dictionary with token usage statistics
            session_id: Optional session identifier
        
        Returns:
            The calculated cost
        """
        # Get model pricing info
        model_info = self.models.get(model, {})
        cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
        
        # Extract token counts
        prompt_tokens = usage_stats.get("prompt_tokens", 0)
        completion_tokens = usage_stats.get("completion_tokens", 0)
        total_tokens = usage_stats.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        
        # Calculate cost
        cost = (total_tokens * cost_per_1k) / 1000
        
        # Log the cost
        cost_logger.info(f"API call logged: model={model}, tokens={total_tokens}, cost=${cost:.6f}")
        
        # Log to console as well
        print(f"Cost tracker: model={model}, tokens={total_tokens}, cost=${cost:.6f}")
        
        return cost
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Mock implementation that returns empty stats"""
        return {
            "session_id": session_id or self.current_session_id,
            "total_cost": 0,
            "total_tokens": 0,
            "calls": 0,
            "models": {}
        }
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Mock implementation that returns empty stats"""
        return {
            "date": date or "today",
            "total_cost": 0,
            "total_tokens": 0,
            "calls": 0,
            "models": {}
        }
    
    def get_cost_trends(self, days: int = 30):
        """Mock implementation that returns empty stats"""
        return {}
    
    def export_cost_report(self, output_file: str = "cost_report.json") -> str:
        """Mock implementation that does nothing"""
        return "Mock export completed"

# Example usage
if __name__ == "__main__":
    # Load config
    import yaml
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        config = {"models": {}}
    
    # Initialize tracker
    tracker = CostTracker(config)
    
    # Simulated API call
    tracker.log_api_call(
        model="anthropic/claude-3-haiku",
        usage_stats={
            "prompt_tokens": 100,
            "completion_tokens": 150,
            "total_tokens": 250
        }
    )
    
    # Print session summary
    print(json.dumps(tracker.get_session_summary(), indent=2))
    
    # Export report
    report_path = tracker.export_cost_report()
    print(f"Cost report exported to: {report_path}") 