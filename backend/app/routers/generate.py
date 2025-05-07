from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging

from ..utils.model_selector import select_model
from ..utils.openrouter import send_request

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_endpoint")

router = APIRouter(
    prefix="/generate",
    tags=["generation"]
)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to generate a response for")
    routing_strategy: str = Field(
        "balanced", 
        description="Strategy for routing: 'balanced', 'cost', 'speed', or 'quality'"
    )
    max_tokens: Optional[int] = Field(1024, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature for generation")
    messages: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation messages")

class GenerateResponse(BaseModel):
    model_used: str
    response: str
    latency_seconds: float
    estimated_cost: float
    routing_explanation: str
    token_metrics: Dict[str, int]

@router.post("/", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Generate a response using the most appropriate AI model based on the prompt and routing strategy.
    """
    start_time = time.time()
    logger.info(f"Generate request received with strategy: {request.routing_strategy}")
    
    try:
        # Select the best model for this prompt
        model_id, explanation = select_model(
            prompt=request.prompt,
            strategy=request.routing_strategy
        )
        
        logger.info(f"Selected model: {model_id}")
        
        # Use provided messages or create a new message list
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        
        # Send the request to the selected model
        response = await send_request(
            model=model_id,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Check if we got an error response
        if isinstance(response, dict) and "error" in response:
            raise HTTPException(
                status_code=500,
                detail=f"Model API error: {response.get('message', 'Unknown error')}"
            )
        
        # Unpack the response tuple
        response_text, usage_stats, latency = response
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Return the formatted response
        return GenerateResponse(
            model_used=model_id,
            response=response_text,
            latency_seconds=total_time,
            estimated_cost=usage_stats.get("cost", 0),
            routing_explanation=explanation,
            token_metrics={
                "prompt": usage_stats.get("prompt_tokens", 0),
                "completion": usage_stats.get("completion_tokens", 0),
                "total": usage_stats.get("total_tokens", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing generate request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        ) 