"""
Task detection module to classify prompts into different task types.
"""
import re
from typing import Literal

TaskType = Literal["text-generation", "code-generation", "summarization"]

def detect_task_type(prompt: str) -> TaskType:
    """
    Detect the type of task based on the prompt.
    
    Args:
        prompt: The user's prompt
        
    Returns:
        Task type classification
    """
    prompt = prompt.lower()
    
    # Check for code-related keywords
    code_indicators = [
        r'(write|create|generate|fix|debug|implement|develop)\s+(a|an|the)?\s*(code|function|class|method|script)',
        r'(python|javascript|typescript|java|c\+\+|ruby|golang|rust|php|html|css|sql)',
        r'(algorithm|function|variable|parameter|return|import|class|object)',
        r'(syntax|compiler|interpreter|runtime|execute)',
        r'```[a-z]*\n'  # Code block markers
    ]
    
    for pattern in code_indicators:
        if re.search(pattern, prompt):
            return "code-generation"
    
    # Check for summarization-related keywords
    summarize_indicators = [
        r'(summarize|summary|summarization|sum up|tl;dr|tldr)',
        r'(condense|shorten|brief|overview|synopsis)',
        r'(key points|main ideas|takeaways|highlights)',
        r'(extract|distill|condense)',
    ]
    
    for pattern in summarize_indicators:
        if re.search(pattern, prompt):
            return "summarization"
    
    # Default to text generation
    return "text-generation" 