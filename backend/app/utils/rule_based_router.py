"""
Enhanced Rule-Based Router for LLM Selection

This module provides a robust rule-based system for routing prompts
to the most appropriate LLM based on prompt type, length, and other
characteristics.
"""

import re
import os
import json
import time
import logging
import yaml
import httpx
import asyncio
from typing import Dict, Any, Tuple, List, Optional, Set
from datetime import datetime

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rule_based_router")

# Add file handler for the router
file_handler = logging.FileHandler(os.path.join(logs_dir, "router.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Constants for prompt classification
CODE_KEYWORDS = {
    'function', 'class', 'def', 'return', 'import', 'from', 'export', 'const', 'let', 'var',
    'async', 'await', 'public', 'private', 'protected', 'interface', 'implements', 'extends',
    'package', 'namespace', 'module', 'library', 'framework', 'algorithm', 'code', 'program', 
    'script', 'syntax', 'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'ruby', 
    'go', 'rust', 'swift', 'kotlin', 'php', 'html', 'css', 'sql', 'bash', 'shell', 'scala'
}

SUMMARY_KEYWORDS = {
    'summarize', 'summarization', 'summary', 'tldr', 'recap', 'overview', 'key points',
    'main ideas', 'digest', 'brief', 'extract', 'condense', 'summarise', 'synopsis',
    'outline', 'gist', 'essence', 'rundown', 'abstract', 'précis'
}

ANALYSIS_KEYWORDS = {
    'analyze', 'analysis', 'examine', 'evaluate', 'assess', 'review', 'breakdown',
    'interpret', 'investigate', 'explore', 'critique', 'compare', 'contrast', 'explain',
    'appraise', 'consider', 'critique', 'dissect', 'understand', 'scrutinize', 'diagnose'
}

CREATIVE_KEYWORDS = {
    'write', 'create', 'design', 'generate', 'compose', 'draft', 'craft', 'story',
    'poem', 'essay', 'article', 'blog', 'post', 'novel', 'fiction', 'creative',
    'imagine', 'invent', 'brainstorm', 'formulate', 'develop', 'produce', 'lyrics',
    'script', 'screenplay', 'narrative', 'tale', 'fantasy', 'sci-fi', 'mystery'
}

QUESTION_STARTERS = {
    'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'can', 'could',
    'would', 'should', 'do', 'does', 'did', 'has', 'have', 'had', 'will', 'may'
}

# Regular expressions for code patterns
CODE_PATTERNS = [
    r'```\w*',                     # Code blocks
    r'function\s+\w+\s*\(',        # Function declarations
    r'class\s+\w+',                # Class declarations
    r'def\s+\w+\s*\(',             # Python function definitions
    r'import\s+\w+',               # Import statements
    r'\bvar\b|\blet\b|\bconst\b',  # JavaScript variables
    r'SELECT.*FROM',               # SQL queries
    r'<\w+>.*</\w+>',              # HTML/XML tags
    r'\[\s*[\w\d_]+\s*:',          # JSON/dictionary notation
    r'^\s*@\w+',                   # Decorators
    r'#include',                   # C/C++ includes
    r'using namespace',            # C++ namespace
    r'public\s+(static\s+)?\w+\s+\w+\s*\(',  # Java/C# method
    r'for\s*\(\s*\w+.+\)',         # For loops
    r'while\s*\(\s*.+\)',          # While loops
    r'if\s*\(\s*.+\)\s*\{',        # If statements
    r'^\s*\/\/ ',                  # Single line comments
    r'^\s*# ',                     # Python/shell comments
    r'^\s*\/\*[\s\S]*?\*\/',       # Multi-line comments
    r'git\s+(clone|pull|push|commit|checkout|merge|rebase|status)',  # Git commands
    r'npm|yarn|pip|conda|apt|brew' # Package managers
]

def load_config():
    """Load configuration from config.yaml file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
        return {"models": {}, "default_model": "anthropic/claude-3-haiku"}

# Global config
CONFIG = load_config()

# Get OpenRouter API key from environment or config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or CONFIG.get("api", {}).get("openrouter_api_key")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

class PromptClassifier:
    """
    Class for classifying prompts into different types based on content analysis.
    Uses rule-based approaches and keyword matching for classification.
    """
    
    def __init__(self):
        """Initialize the classifier with patterns from config"""
        self.prompt_types = CONFIG.get("prompt_types", {})
        # Set confidence threshold for using GPT-4 fallback
        self.confidence_threshold = 2  # Minimum keyword matches for high confidence
        self.use_gpt4_fallback = CONFIG.get("classification", {}).get("use_gpt4_fallback", True)
        
    def get_token_set(self, text: str) -> Set[str]:
        """
        Extract a set of normalized tokens from text.
        
        Args:
            text: The input text
            
        Returns:
            Set of lowercase tokens
        """
        # Simple tokenization - split by non-alphanumeric and convert to lowercase
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)
    
    def count_matches(self, tokens: Set[str], keywords: Set[str]) -> int:
        """
        Count how many keywords appear in the tokens.
        
        Args:
            tokens: Set of tokens from the text
            keywords: Set of keywords to match
            
        Returns:
            Number of matching keywords
        """
        return len(tokens.intersection(keywords))
    
    def has_code_pattern(self, text: str) -> bool:
        """
        Check if the text contains code patterns.
        
        Args:
            text: The input text
            
        Returns:
            True if code patterns are found
        """
        for pattern in CODE_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def is_question(self, text: str) -> bool:
        """
        Check if the text is a question.
        
        Args:
            text: The input text
            
        Returns:
            True if the text is likely a question
        """
        # Check if ends with question mark
        if text.strip().endswith('?'):
            return True
        
        # Check if starts with question words
        first_word = text.strip().split(' ')[0].lower()
        return first_word in QUESTION_STARTERS
    
    async def classify_with_gpt4(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Use GPT-4 to classify the prompt when rule-based classification has low confidence.
        
        Args:
            text: The prompt text
            
        Returns:
            Tuple of (prompt_type, details)
        """
        if not OPENROUTER_API_KEY:
            logger.warning("OpenRouter API key not available. Cannot use GPT-4 fallback for classification.")
            return "general", {"method": "default_no_api_key"}
            
        logger.info("Using GPT-4 to classify prompt with low confidence")
        
        # Prepare the system message for GPT-4
        system_prompt = """
        Classify the user's prompt into exactly ONE of these categories:
        - code (programming, development, debugging)
        - summary (summarizing, condensing information)
        - analysis (analyzing, evaluating, comparing)
        - creative (writing creative content, stories)
        - question (simple question answering)
        - general (anything else)
        
        Respond with ONLY the category name, no other text.
        """
        
        # Prepare the messages for GPT-4
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:1000]}  # Truncate to avoid token limits
        ]
        
        try:
            # Make the API call to OpenRouter for GPT-4
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "openai/gpt-4",
                        "messages": messages,
                        "max_tokens": 20,
                        "temperature": 0.1
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Error calling GPT-4 for classification: {response.status_code} - {response.text}")
                    return "general", {"method": "default_api_error"}
                    
                result = response.json()
                classification = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                
                # Validate the classification is one of our expected categories
                valid_types = ["code", "summary", "analysis", "creative", "question", "general"]
                if classification not in valid_types:
                    logger.warning(f"GPT-4 returned unexpected classification: {classification}")
                    return "general", {"method": "default_invalid_gpt4_response"}
                
                logger.info(f"GPT-4 classified prompt as: {classification}")
                return classification, {"method": "gpt4_fallback", "confidence": "high"}
                
        except Exception as e:
            logger.error(f"Exception using GPT-4 for classification: {str(e)}")
            return "general", {"method": "default_exception", "error": str(e)}
    
    async def classify(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify the text into a prompt type with detailed analysis.
        This is the async version that can use GPT-4 fallback.
        
        Args:
            text: The prompt text
            
        Returns:
            Tuple of (prompt_type, details)
        """
        # First check config patterns
        for prompt_type, type_config in self.prompt_types.items():
            patterns = type_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.debug(f"Classified as {prompt_type} based on config pattern: {pattern}")
                    return prompt_type, {"method": "config_pattern", "pattern": pattern, "confidence": "high"}
        
        # Process text to get tokens
        tokens = self.get_token_set(text)
        
        # Initialize match counts
        matches = {
            "code": self.count_matches(tokens, CODE_KEYWORDS),
            "summary": self.count_matches(tokens, SUMMARY_KEYWORDS),
            "analysis": self.count_matches(tokens, ANALYSIS_KEYWORDS),
            "creative": self.count_matches(tokens, CREATIVE_KEYWORDS),
        }
        
        # Check for code patterns (stronger signal)
        if self.has_code_pattern(text):
            matches["code"] += 10  # Heavily weight code patterns
        
        # Check if it's a question
        is_question = self.is_question(text)
        
        # Determine the type based on highest match count
        max_type = max(matches, key=matches.get)
        max_count = matches[max_type]
        
        # If it's a strong match, use that type
        if max_count >= self.confidence_threshold:
            logger.debug(f"Classified as {max_type} based on keyword matches ({max_count})")
            return max_type, {"method": "keyword", "matches": matches, "confidence": "high"}
        
        # If it's a question with no strong keyword matches, classify as question
        if is_question:
            logger.debug("Classified as question based on question structure")
            return "question", {"method": "question_structure", "is_question": True, "confidence": "medium"}
        
        # Low confidence case - use GPT-4 if enabled
        if self.use_gpt4_fallback:
            logger.info("Low confidence in rule-based classification, using GPT-4 fallback")
            return await self.classify_with_gpt4(text)
        
        # Default to general if no strong signal and GPT-4 fallback not used
        logger.debug("Classified as general (no strong signals)")
        return "general", {"method": "default", "matches": matches, "confidence": "low"}
        
    def classify_sync(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Synchronous version of classify that doesn't use GPT-4 fallback.
        Provided for backward compatibility.
        
        Args:
            text: The prompt text
            
        Returns:
            Tuple of (prompt_type, details)
        """
        # First check config patterns
        for prompt_type, type_config in self.prompt_types.items():
            patterns = type_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.debug(f"Classified as {prompt_type} based on config pattern: {pattern}")
                    return prompt_type, {"method": "config_pattern", "pattern": pattern}
        
        # Process text to get tokens
        tokens = self.get_token_set(text)
        
        # Initialize match counts
        matches = {
            "code": self.count_matches(tokens, CODE_KEYWORDS),
            "summary": self.count_matches(tokens, SUMMARY_KEYWORDS),
            "analysis": self.count_matches(tokens, ANALYSIS_KEYWORDS),
            "creative": self.count_matches(tokens, CREATIVE_KEYWORDS),
        }
        
        # Check for code patterns (stronger signal)
        if self.has_code_pattern(text):
            matches["code"] += 10  # Heavily weight code patterns
        
        # Check if it's a question
        is_question = self.is_question(text)
        
        # Determine the type based on highest match count
        max_type = max(matches, key=matches.get)
        max_count = matches[max_type]
        
        # If it's a strong match, use that type
        if max_count >= 2:
            logger.debug(f"Classified as {max_type} based on keyword matches ({max_count})")
            return max_type, {"method": "keyword", "matches": matches}
        
        # If it's a question with no strong keyword matches, classify as question
        if is_question:
            logger.debug("Classified as question based on question structure")
            return "question", {"method": "question_structure", "is_question": True}
        
        # Default to general if no strong signal
        logger.debug("Classified as general (no strong signals)")
        return "general", {"method": "default", "matches": matches}

class EnhancedRuleBasedRouter:
    """
    Enhanced rule-based router for selecting LLMs based on prompt analysis.
    """
    
    def __init__(self):
        """Initialize the router with configuration and classifier"""
        self.config = CONFIG
        self.classifier = PromptClassifier()
        self.default_model = self.config.get("default_model", "anthropic/claude-3-haiku")
        
        # Set up the logger directory
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "model_calls")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token for English text
        return len(text) // 4
    
    def categorize_length(self, token_count: int) -> str:
        """
        Categorize the prompt length.
        
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
    
    def extract_requested_model(self, text: str) -> Optional[str]:
        """
        Check if text explicitly requests a specific model.
        
        Args:
            text: The prompt text
            
        Returns:
            Model ID if explicitly requested, None otherwise
        """
        # Check for patterns like "use GPT-4" or "using Claude"
        model_mentions = {
            r'\bgpt-?4\b': "openai/gpt-4",
            r'\bgpt-?4o\b': "openai/gpt-4",
            r'\bclaude-?3\b': "anthropic/claude-3-opus",
            r'\bclaude-?3-opus\b': "anthropic/claude-3-opus",
            r'\bclaude-?3-sonnet\b': "anthropic/claude-3-sonnet",
            r'\bclaude-?3-haiku\b': "anthropic/claude-3-haiku",
            r'\bclaude\b': "anthropic/claude-3-opus",
            r'\bmixtral\b': "mistralai/mixtral-8x7b-instruct",
            r'\bmistral\b': "mistralai/mistral-7b-instruct",
            r'\bllama\b': "meta-llama/llama-2-70b-chat"
        }
        
        for pattern, model_id in model_mentions.items():
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"User explicitly requested model: {model_id}")
                return model_id
        
        return None
    
    def get_model_from_rules(self, prompt_type: str, length_category: str) -> Optional[str]:
        """
        Get model from rule-based configuration based on prompt type and length.
        
        Args:
            prompt_type: Type of prompt (code, summary, question, etc.)
            length_category: Length category (short, medium, long)
            
        Returns:
            Model ID if found in rules, None otherwise
        """
        # Map our prompt types to the rule_based_router categories
        rule_based_type = "question_models"  # Default
        
        if prompt_type == "code":
            rule_based_type = "code_models"
        elif prompt_type in ["summary", "analysis"]:
            rule_based_type = "summary_models"
        elif prompt_type in ["creative", "question"]:
            rule_based_type = "question_models"
        
        # Try to get the model from rule-based router based on prompt type and length
        rule_based = self.config.get("rule_based_router", {})
        rule_based_models = rule_based.get(rule_based_type, {})
        
        return rule_based_models.get(length_category)
    
    def log_model_selection(self, prompt: str, model_id: str, data: Dict[str, Any]) -> None:
        """
        Log detailed information about model selection to a JSON file.
        
        Args:
            prompt: The original prompt
            model_id: The selected model ID
            data: Additional data about the selection
        """
        # Create a log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:1000],  # Truncate very long prompts
            "model_selected": model_id,
            "prompt_type": data.get("prompt_type", "unknown"),
            "length_category": data.get("length_category", "unknown"),
            "token_count": data.get("token_count", 0),
            "strategy": data.get("strategy", "balanced"),
            "user_requested": data.get("user_requested", False),
            "classification_details": data.get("classification_details", {}),
            "selection_reason": data.get("reason", "unknown")
        }
        
        # Create a unique filename based on timestamp
        filename = f"selection_{int(time.time())}_{hash(prompt) % 10000}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Write to file
        try:
            with open(filepath, "w") as f:
                json.dump(log_entry, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging model selection: {e}")
    
    async def select_model_async(self, prompt: str, strategy: str = "balanced") -> Tuple[str, str, Dict[str, Any]]:
        """
        Async version of select_model that can use GPT-4 fallback for classification.
        
        Args:
            prompt: The user's prompt
            strategy: The routing strategy to use
            
        Returns:
            Tuple of (model_id, reason, details)
        """
        start_time = time.time()
        logger.info(f"Selecting model for prompt with strategy: {strategy}")
        
        # Initialize selection details
        details = {
            "strategy": strategy,
            "processing_time": 0,
            "user_requested": False
        }
        
        # Check if user explicitly requested a model
        requested_model = self.extract_requested_model(prompt)
        if requested_model:
            details["user_requested"] = True
            reason = f"Selected {requested_model} as explicitly requested by user"
            
            # Calculate processing time
            details["processing_time"] = time.time() - start_time
            
            # Log the selection
            self.log_model_selection(prompt, requested_model, {
                **details,
                "reason": reason
            })
            
            return requested_model, reason, details
        
        # Analyze the prompt - async version with possible GPT-4 fallback
        prompt_type, classification_details = await self.classifier.classify(prompt)
        token_count = self.estimate_tokens(prompt)
        length_category = self.categorize_length(token_count)
        
        # Store analysis results in details
        details.update({
            "prompt_type": prompt_type,
            "token_count": token_count,
            "length_category": length_category,
            "classification_details": classification_details
        })
        
        # Check for any strategy-specific default model in config
        strategy_config = self.config.get("routing_strategies", {}).get(strategy, {})
        strategy_default = strategy_config.get("default_model")
        
        # Get model from rule-based router
        model_from_rules = self.get_model_from_rules(prompt_type, length_category)
        
        # Get preferred model from prompt type config
        prompt_type_config = self.config.get("prompt_types", {}).get(prompt_type, {})
        preferred_model = prompt_type_config.get("preferred_model")
        
        # Decision logic based on strategy
        model_id = None
        reason = None
        
        if strategy == "cost":
            # Prioritize cost - use rule-based recommendation first, then strategy default
            if length_category == "short":
                model_id = "anthropic/claude-3-haiku"  # Cheapest for short prompts
            elif prompt_type == "code" and length_category != "long":
                model_id = "mistralai/mistral-7b-instruct"  # Good balance for medium coding
            else:
                model_id = model_from_rules or strategy_default or self.default_model
            
            reason = f"Selected {model_id} for cost optimization with {prompt_type} task ({length_category} length)"
            
        elif strategy == "speed":
            # Prioritize speed - smaller/faster models
            if prompt_type == "code" and length_category != "long":
                model_id = "openai/gpt-3.5-turbo"  # Fast for coding
            elif length_category == "short":
                model_id = "anthropic/claude-3-haiku"  # Fastest for short tasks
            else:
                model_id = model_from_rules or "anthropic/claude-3-haiku"  # Balance speed
            
            reason = f"Selected {model_id} for speed optimization with {prompt_type} task ({length_category} length)"
            
        elif strategy == "quality":
            # Prioritize quality - use the most capable models
            if prompt_type == "code":
                model_id = "openai/gpt-4"  # Best for code
            elif prompt_type in ["summary", "analysis"] and length_category == "long":
                model_id = "anthropic/claude-3-opus"  # Best for long analysis
            else:
                model_id = "anthropic/claude-3-sonnet"  # Good balance of quality
            
            reason = f"Selected {model_id} for quality optimization with {prompt_type} task"
            
        else:  # balanced strategy or unknown strategy
            # Balanced approach - consider task type and length
            if prompt_type == "code":
                if length_category == "long":
                    model_id = "openai/gpt-4"
                else:
                    model_id = "openai/gpt-3.5-turbo"
            elif prompt_type in ["summary", "analysis"]:
                if length_category == "long":
                    model_id = "anthropic/claude-3-sonnet"
                else:
                    model_id = "anthropic/claude-3-haiku"
            elif prompt_type == "creative":
                model_id = "anthropic/claude-3-sonnet"  # Good for creative tasks
            else:
                # Default to rules or preferred model
                model_id = model_from_rules or preferred_model or self.default_model
            
            reason = f"Selected {model_id} for balanced performance with {prompt_type} task ({length_category} length)"
        
        # Fallback to default if no model was selected
        if not model_id:
            model_id = self.default_model
            reason = f"Falling back to default model {model_id}"
        
        # Calculate processing time
        details["processing_time"] = time.time() - start_time
        
        # Log the selection
        self.log_model_selection(prompt, model_id, {
            **details,
            "reason": reason
        })
        
        return model_id, reason, details
        
    def select_model(self, prompt: str, strategy: str = "balanced") -> Tuple[str, str, Dict[str, Any]]:
        """
        Select the most appropriate model based on prompt analysis and strategy.
        This is the synchronous version that doesn't use GPT-4 fallback.
        
        Args:
            prompt: The user's prompt
            strategy: The routing strategy to use
            
        Returns:
            Tuple of (model_id, reason, details)
        """
        start_time = time.time()
        logger.info(f"Selecting model for prompt with strategy: {strategy}")
        
        # Initialize selection details
        details = {
            "strategy": strategy,
            "processing_time": 0,
            "user_requested": False
        }
        
        # Check if user explicitly requested a model
        requested_model = self.extract_requested_model(prompt)
        if requested_model:
            details["user_requested"] = True
            reason = f"Selected {requested_model} as explicitly requested by user"
            
            # Calculate processing time
            details["processing_time"] = time.time() - start_time
            
            # Log the selection
            self.log_model_selection(prompt, requested_model, {
                **details,
                "reason": reason
            })
            
            return requested_model, reason, details
        
        # Analyze the prompt - use the sync version
        prompt_type, classification_details = self.classifier.classify_sync(prompt)
        token_count = self.estimate_tokens(prompt)
        length_category = self.categorize_length(token_count)
        
        # Store analysis results in details
        details.update({
            "prompt_type": prompt_type,
            "token_count": token_count,
            "length_category": length_category,
            "classification_details": classification_details
        })
        
        # Check for any strategy-specific default model in config
        strategy_config = self.config.get("routing_strategies", {}).get(strategy, {})
        strategy_default = strategy_config.get("default_model")
        
        # Get model from rule-based router
        model_from_rules = self.get_model_from_rules(prompt_type, length_category)
        
        # Get preferred model from prompt type config
        prompt_type_config = self.config.get("prompt_types", {}).get(prompt_type, {})
        preferred_model = prompt_type_config.get("preferred_model")
        
        # Decision logic based on strategy
        model_id = None
        reason = None
        
        if strategy == "cost":
            # Prioritize cost - use rule-based recommendation first, then strategy default
            if length_category == "short":
                model_id = "anthropic/claude-3-haiku"  # Cheapest for short prompts
            elif prompt_type == "code" and length_category != "long":
                model_id = "mistralai/mistral-7b-instruct"  # Good balance for medium coding
            else:
                model_id = model_from_rules or strategy_default or self.default_model
            
            reason = f"Selected {model_id} for cost optimization with {prompt_type} task ({length_category} length)"
            
        elif strategy == "speed":
            # Prioritize speed - smaller/faster models
            if prompt_type == "code" and length_category != "long":
                model_id = "openai/gpt-3.5-turbo"  # Fast for coding
            elif length_category == "short":
                model_id = "anthropic/claude-3-haiku"  # Fastest for short tasks
            else:
                model_id = model_from_rules or "anthropic/claude-3-haiku"  # Balance speed
            
            reason = f"Selected {model_id} for speed optimization with {prompt_type} task ({length_category} length)"
            
        elif strategy == "quality":
            # Prioritize quality - use the most capable models
            if prompt_type == "code":
                model_id = "openai/gpt-4"  # Best for code
            elif prompt_type in ["summary", "analysis"] and length_category == "long":
                model_id = "anthropic/claude-3-opus"  # Best for long analysis
            else:
                model_id = "anthropic/claude-3-sonnet"  # Good balance of quality
            
            reason = f"Selected {model_id} for quality optimization with {prompt_type} task"
            
        else:  # balanced strategy or unknown strategy
            # Balanced approach - consider task type and length
            if prompt_type == "code":
                if length_category == "long":
                    model_id = "openai/gpt-4"
                else:
                    model_id = "openai/gpt-3.5-turbo"
            elif prompt_type in ["summary", "analysis"]:
                if length_category == "long":
                    model_id = "anthropic/claude-3-sonnet"
                else:
                    model_id = "anthropic/claude-3-haiku"
            elif prompt_type == "creative":
                model_id = "anthropic/claude-3-sonnet"  # Good for creative tasks
            else:
                # Default to rules or preferred model
                model_id = model_from_rules or preferred_model or self.default_model
            
            reason = f"Selected {model_id} for balanced performance with {prompt_type} task ({length_category} length)"
        
        # Fallback to default if no model was selected
        if not model_id:
            model_id = self.default_model
            reason = f"Falling back to default model {model_id}"
        
        # Calculate processing time
        details["processing_time"] = time.time() - start_time
        
        # Log the selection
        self.log_model_selection(prompt, model_id, {
            **details,
            "reason": reason
        })
        
        return model_id, reason, details

async def select_model_async(prompt: str, strategy: str = "balanced") -> Tuple[str, str]:
    """
    Async convenience function to select a model using the EnhancedRuleBasedRouter.
    This version can use GPT-4 fallback for classification when confidence is low.
    
    Args:
        prompt: The user's prompt
        strategy: The routing strategy to use
        
    Returns:
        Tuple of (model_id, reason)
    """
    router = EnhancedRuleBasedRouter()
    model_id, reason, _ = await router.select_model_async(prompt, strategy)
    return model_id, reason

def select_model(prompt: str, strategy: str = "balanced") -> Tuple[str, str]:
    """
    Convenience function to select a model using the EnhancedRuleBasedRouter.
    This is the synchronous version that doesn't use GPT-4 fallback.
    
    Args:
        prompt: The user's prompt
        strategy: The routing strategy to use
        
    Returns:
        Tuple of (model_id, reason)
    """
    router = EnhancedRuleBasedRouter()
    model_id, reason, _ = router.select_model(prompt, strategy)
    return model_id, reason

# For testing
if __name__ == "__main__":
    router = EnhancedRuleBasedRouter()
    
    test_prompts = [
        "Write a Python function to calculate Fibonacci numbers",
        "Summarize the main points of climate change research from the last decade",
        "What is the capital of France?",
        "Write a short poem about the sunset",
        "Analyze the impact of artificial intelligence on modern society",
        "function calculateTotal(items) {\n  return items.reduce((sum, item) => sum + item.price, 0);\n}",
        "```python\ndef hello_world():\n    print('Hello, world!')\n```",
        "Compare and contrast quantum computing with classical computing",
        "How do I fix this error: TypeError: Cannot read property 'map' of undefined"
    ]
    
    async def run_tests():
        print("Testing rule-based router with different prompts and strategies:")
        print("-" * 80)
        
        for prompt in test_prompts:
            print(f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}")
            
            for strategy in ["balanced", "cost", "speed", "quality"]:
                # Test async version with GPT-4 fallback
                model_id, reason, details = await router.select_model_async(prompt, strategy)
                print(f"  Strategy: {strategy} (with GPT-4 fallback)")
                print(f"  Selected: {model_id}")
                print(f"  Reason: {reason}")
                print(f"  Type: {details['prompt_type']}, Length: {details['length_category']} ({details['token_count']} tokens)")
                print(f"  Classification method: {details.get('classification_details', {}).get('method', 'unknown')}")
                print(f"  Processing time: {details['processing_time']*1000:.2f}ms")
                print()
            
            print("-" * 80)
    
    if OPENROUTER_API_KEY:
        asyncio.run(run_tests())
    else:
        print("OpenRouter API key not found. Running sync tests only.")
        for prompt in test_prompts:
            print(f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}")
            
            for strategy in ["balanced", "cost", "speed", "quality"]:
                model_id, reason, details = router.select_model(prompt, strategy)
                print(f"  Strategy: {strategy}")
                print(f"  Selected: {model_id}")
                print(f"  Reason: {reason}")
                print(f"  Type: {details['prompt_type']}, Length: {details['length_category']} ({details['token_count']} tokens)")
                print(f"  Processing time: {details['processing_time']*1000:.2f}ms")
                print()
            
            print("-" * 80) 