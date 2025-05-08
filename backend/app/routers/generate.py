from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import json
import logging
import asyncio
from random import uniform, choice

from ..utils.rule_based_router import select_model_async, select_model
from ..utils.openrouter_client import chat, ask

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_endpoint")

router = APIRouter(
    prefix="/generate",
    tags=["generation"]
)

# For tracking request timing to prevent rate limits
last_request_time = 0

# Define model groups for rotation
MODEL_GROUPS = {
    "anthropic": ["anthropic/claude-3-opus", "anthropic/claude-3-sonnet", "anthropic/claude-3-haiku"],
    "openai": ["openai/gpt-4", "openai/gpt-3.5-turbo"],
    "mistral": ["mistralai/mixtral-8x7b-instruct", "mistralai/mistral-7b-instruct"]
}

# Track last used models to avoid using the same one twice in a row
last_used_models = set()

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to generate a response for")
    routing_strategy: str = Field(
        "balanced", 
        description="Strategy for routing: 'balanced', 'cost', 'speed', or 'quality'"
    )
    max_tokens: Optional[int] = Field(1024, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature for generation")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages")
    use_gpt4_classification: Optional[bool] = Field(True, description="Whether to use GPT-4 for classification when confidence is low")

class GenerateResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    tokens: int # Total tokens
    cost: float
    latency: float # Total latency for the request in seconds
    classification: Dict[str, Any] = Field(default_factory=dict)

def get_alternative_model(preferred_model: str) -> str:
    """
    Get an alternative model from a different provider to avoid rate limiting
    """
    global last_used_models
    
    # Find which provider the preferred model belongs to
    provider = None
    for p, models in MODEL_GROUPS.items():
        if preferred_model in models:
            provider = p
            break
    
    # If we can't determine the provider, choose a random model from any provider
    if not provider:
        all_models = [m for models_list in MODEL_GROUPS.values() for m in models_list] # Corrected variable name
        # Filter out recently used models
        available_models = [m for m in all_models if m not in last_used_models and m != preferred_model]
        # If no available models, reset and use any model except the preferred one
        if not available_models:
            available_models = [m for m in all_models if m != preferred_model]
        
        chosen_model = choice(available_models) if available_models else "anthropic/claude-3-haiku"
    else:
        # Choose an alternative provider
        alternative_providers = [p for p in MODEL_GROUPS.keys() if p != provider]
        if not alternative_providers:
            alternative_providers = list(MODEL_GROUPS.keys())
        
        alt_provider = choice(alternative_providers)
        # Choose a model from the alternative provider
        available_models = [m for m in MODEL_GROUPS[alt_provider] if m not in last_used_models]
        if not available_models:
            available_models = MODEL_GROUPS[alt_provider]
        
        chosen_model = choice(available_models)
    
    # Update last used models (keep track of last 3)
    # Make sure last_used_models is treated as a set for adding, then convert to list for popping if needed
    if not isinstance(last_used_models, set):
        last_used_models = set(last_used_models)
    last_used_models.add(chosen_model)
    if len(last_used_models) > 3:
        # To pop from a set, typically you remove an arbitrary element or a specific one if known.
        # If order matters, consider a list or deque. For now, removing an arbitrary one if size exceeds 3.
        last_used_models.pop() # This will remove an arbitrary element from the set
    
    logger.info(f"Selected alternative model: {chosen_model} (original was {preferred_model})")
    return chosen_model

@router.post("/", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text response for a given prompt or chat messages.
    Routes to the most appropriate model based on prompt type and routing strategy.
    """
    global last_request_time
    
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    # Add a small delay between requests to avoid rate limits
    time_since_last = start_time - last_request_time
    if last_request_time > 0 and time_since_last < 0.8:
        delay_time = 0.8 - time_since_last + uniform(0.1, 0.3)
        logger.info(f"[{request_id}] Adding {delay_time:.2f}s delay between requests to prevent rate limits")
        await asyncio.sleep(delay_time)
    
    # Update last request time
    last_request_time = time.time()
    
    try:
        # Model selection is based on the current prompt
        model_id, reason = await select_model_async(request.prompt, request.routing_strategy)
        
        # Occasionally rotate models to avoid hitting rate limits with the same provider
        if uniform(0, 1) < 0.3:  # 30% chance to use an alternative model
            original_model = model_id
            model_id = get_alternative_model(model_id)
            reason += f" (rotated from {original_model} to avoid rate limits)"
        
        logger.info(f"[{request_id}] Selected model {model_id}: {reason}")

        # Prepare messages for the API call
        api_messages: List[Dict[str, Any]]
        if request.messages:
            # If history is provided, append the current prompt to it
            api_messages = list(request.messages) # Create a mutable copy
            api_messages.append({"role": "user", "content": request.prompt})
        else:
            # If no history, the current prompt is the only message
            api_messages = [{"role": "user", "content": request.prompt}]
        
        # Generate response using the chat function, as it's more general
        response = await chat(
            messages=api_messages,
            model=model_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        if not response.get("success", False):
            error_detail = response.get("error", "Unknown error")
            logger.warning(f"[{request_id}] Primary model {model_id} failed: {error_detail}. Trying fallback model.")
            
            fallback_model = get_alternative_model(model_id)
            
            response = await chat( # Always use chat for consistency with api_messages
                messages=api_messages, # Use the same full message list for fallback
                model=fallback_model,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            if not response.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=f"Both primary and fallback models failed. Error: {response.get('error')}"
                )
        
        total_latency = time.time() - start_time
        
        return {
            "text": response["text"],
            "model": response["model"],
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "tokens": response["tokens"],
            "cost": response["cost"],
            "latency": total_latency,
            "classification": {
                "prompt_type": response.get("prompt_type", "unknown"),
                "selected_reason": reason,
                "tried_models": response.get("tried_models", [model_id])
            }
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error generating response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        ) 