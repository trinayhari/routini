from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from ..utils.openrouter import send_request
import asyncio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ModelMetadata(BaseModel):
    latencyMs: int
    totalTokens: int
    promptTokens: int
    completionTokens: int
    cost: float
    model: str

class ModelResponse(BaseModel):
    model: str
    response: str
    metadata: ModelMetadata

class CompareRequest(BaseModel):
    prompt: str
    models: Optional[List[str]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class CompareResponse(BaseModel):
    results: List[ModelResponse]

@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """
    Compare responses from multiple models for the same prompt.
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Use default models if none specified
    if not request.models:
        request.models = [
            "openai/gpt-4",
            "anthropic/claude-3-opus",
            "mistralai/mixtral-8x7b-instruct"
        ]
    
    try:
        # Send requests to all models in parallel
        tasks = []
        for model in request.models:
            task = send_request(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        results = []
        for model, response in zip(request.models, responses):
            if isinstance(response, Exception):
                logger.error(f"Error from {model}: {str(response)}")
                results.append(ModelResponse(
                    model=model,
                    response=f"Error: {str(response)}",
                    metadata=ModelMetadata(
                        latencyMs=0,
                        totalTokens=0,
                        promptTokens=0,
                        completionTokens=0,
                        cost=0.0,
                        model=model
                    )
                ))
            else:
                response_text, usage_stats, latency = response
                results.append(ModelResponse(
                    model=model,
                    response=response_text,
                    metadata=ModelMetadata(
                        latencyMs=int(latency * 1000),  # Convert to milliseconds
                        totalTokens=usage_stats["total_tokens"],
                        promptTokens=usage_stats["prompt_tokens"],
                        completionTokens=usage_stats["completion_tokens"],
                        cost=float(usage_stats["cost"]),
                        model=model
                    )
                ))
        
        return CompareResponse(results=results)
        
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 