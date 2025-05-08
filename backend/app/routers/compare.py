from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from ..utils.openrouter_client import compare
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
    request_id = f"req_{int(id(request))}"  # Generate a unique ID for this request
    
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
        logger.info(f"[{request_id}] Starting model comparison with {len(request.models)} models")
        
        # Use our new compare function from the OpenRouter client
        comparison = await compare(
            prompt=request.prompt,
            models=request.models,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Process the comparison results
        results = []
        for result in comparison["results"]:
            # Safely get the model name, providing a default if it's missing
            model_name = result.get("model", "unknown_model_on_error")
            
            if result.get("success", False):
                # Successful response
                results.append(ModelResponse(
                    model=model_name, # Use the safely retrieved model_name
                    response=result.get("text", ""),
                    metadata=ModelMetadata(
                        latencyMs=int(result.get("latency", 0) * 1000),  # Convert to milliseconds
                        totalTokens=result.get("tokens", 0),
                        promptTokens=result.get("prompt_tokens", 0),
                        completionTokens=result.get("completion_tokens", 0),
                        cost=float(result.get("cost", 0.0)),
                        model=model_name # Use the safely retrieved model_name
                    )
                ))
            else:
                # Error response
                results.append(ModelResponse(
                    model=model_name, # Use the safely retrieved model_name
                    response=f"Error: {result.get('error', 'Unknown error')}",
                    metadata=ModelMetadata(
                        latencyMs=int(result.get("latency", 0) * 1000),
                        totalTokens=0,
                        promptTokens=0,
                        completionTokens=0,
                        cost=0.0,
                        model=model_name # Use the safely retrieved model_name
                    )
                ))
        
        logger.info(f"[{request_id}] Comparison completed in {comparison.get('total_latency', 0.0):.2f}s")
        return CompareResponse(results=results)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error comparing models: {str(e)}")
        # Propagate the original error detail if it's a KeyError for 'model'
        if isinstance(e, KeyError) and str(e) == "'model'":
             raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model comparison: {str(e)}") 