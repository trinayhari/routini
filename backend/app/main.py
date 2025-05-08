from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
import logging
import time
import yaml
import os
import json
from contextlib import asynccontextmanager

from .routers import generate, compare

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("app")

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
        return {"app": {"name": "AI Model Router API"}}

config = load_config()

# Lifespan event handler for resource management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load any resources or perform initialization
    logger.info("Starting AI Model Router API")
    yield
    # Shutdown: Close any resources or perform cleanup
    logger.info("Shutting down AI Model Router API")

# Create the FastAPI application
app = FastAPI(
    title=config.get("app", {}).get("name", "AI Model Router API"),
    description="API for routing AI model requests to the most appropriate model",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Special handler for validation errors (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors and return a more helpful response.
    This is especially useful for handling issues with metadata serialization.
    """
    # Get the body that was sent
    body = await request.body()
    try:
        body_json = json.loads(body)
    except:
        body_json = {"raw": str(body)}
    
    # Log the details
    logger.error(f"Validation error: {exc}")
    logger.error(f"Request body: {body_json}")
    
    # Check for specific metadata issues
    if any("metadata" in err["loc"] for err in exc.errors()):
        # Log all errors related to metadata
        metadata_errors = [err for err in exc.errors() if "metadata" in err["loc"]]
        logger.error(f"Metadata-related errors: {metadata_errors}")
        
        # Create a helpful response
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Metadata formatting error",
                "message": "The metadata field should be a JSON string or a valid object. If sending from frontend, ensure metadata is properly serialized.",
                "errors": exc.errors(),
                "help": "Try using JSON.stringify() on any metadata objects before sending."
            },
        )
    
    # For other validation errors, return the standard response
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body_json},
    )

# Add logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Generate a unique request ID
    request_id = f"{int(start_time)}-{os.urandom(4).hex()}"
    request.state.request_id = request_id
    
    # Log the incoming request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Log the response
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed - Status: {response.status_code}, Time: {process_time:.3f}s")
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        # Log and handle any exceptions
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed - Error: {str(e)}, Time: {process_time:.3f}s")
        
        return Response(
            content={"error": "Internal server error", "detail": str(e)},
            status_code=500,
            media_type="application/json"
        )

# Include routers
app.include_router(generate.router)
app.include_router(compare.router)

# Root endpoint for health check
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AI Model Router API is running",
        "version": "1.0.0",
    }

# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 