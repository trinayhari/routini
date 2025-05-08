# Model Router API

FastAPI backend that routes requests to the best AI model based on task type and routing strategy using OpenRouter API.

## Features

- Task type detection based on prompt (text-generation, code-generation, summarization)
- Multiple routing strategies (fastest, cheapest, most_capable, balanced)
- Model selection from a configurable set of models in `config.yaml`
- Cost and latency tracking

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the `env.example` file to `.env` and add your OpenRouter API key:

```bash
cp env.example .env
# Edit .env with your API key
```

## Running the Server

Start the FastAPI server:

```bash
cd backend
uvicorn app.main:app --reload
```

The server will be available at http://localhost:8000

## API Documentation

API documentation is automatically generated and available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Usage Example

```bash
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about AI",
    "routing_strategy": "balanced",
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

Response:

```json
{
  "model_used": "anthropic/claude-3-sonnet-20240229",
  "response": "Silicon dreams in neural nets unfold,\nPatterns learned from data new and old.\nA dance of math, of logic, and of light,\nArtificial minds reaching new height.\n\nBeyond the code, a question lingers still:\nCan circuits know the depths of human will?",
  "latency_seconds": 3.24,
  "estimated_cost": 0.00075,
  "routing_explanation": "Selected a balanced model for text-generation considering both performance and cost. Model latency: 6.5 seconds, Cost: $0.003 per 1K tokens."
}
```

## Configuration

The model configurations are stored in `config.yaml` in the root directory. You can add or modify models as needed.

## Error Handling

The OpenRouter integration has been improved with robust error handling:

1. **Standardized Error Classification** - Errors are now categorized into specific types:

   - `NetworkError` - Connection issues, timeouts, server errors
   - `AuthenticationError` - API key issues, authentication failures
   - `RateLimitError` - When rate limits are exceeded
   - `ModelError` - Issues with model parameters or prompt content
   - `UnknownAPIError` - Other API errors

2. **Automatic Retries** - Network errors and rate limit errors are automatically retried with exponential backoff.

3. **Empty Response Handling** - The system now properly detects and handles blank responses from models, providing helpful fallback messages instead of empty outputs.

4. **Input Validation** - Improved validation of prompts and message content to prevent issues before they occur.

5. **Metadata Handling** - Support for both string and object metadata formats, with proper conversion between formats.

6. **Detailed Error Responses** - All error responses now include detailed information about what went wrong and how to fix it.

## Running Tests

To verify error handling, run the test suite:

```bash
./run_tests.sh
```

This will:

1. Test the error handling with various error scenarios
2. Verify the client integration with API endpoints
3. Validate metadata handling and input validation

## Troubleshooting

If you encounter errors:

1. **422 Validation Errors**: Usually related to metadata formatting. Ensure metadata is either a valid JSON string or an object.
2. **Empty Responses**: If you get blank outputs or error messages like "The model returned an empty response", you can:

   - Run the diagnostic tool to troubleshoot:

     ```bash
     cd backend
     ./diagnose_empty_responses.py --models anthropic/claude-3-opus --prompt "Your problematic prompt"
     ```

   - Check the generated `diagnosis_results.json` file for detailed diagnostics

   - Try using a different model (e.g., switch from claude-3-opus to claude-3-sonnet)

   - Check if your prompt might be triggering content filters

   - Verify your API key and quota with OpenRouter

3. **Rate Limiting**: If you see rate limit errors, implement request throttling or switch to models with higher throughput.

## Diagnostics

The system includes diagnostic tools to help troubleshoot issues:

1. **Error Handler**: The `error_handler.py` module provides standardized error handling with proper classification and retries.

2. **Empty Response Diagnosis**: The `diagnose_empty_responses.py` script tests multiple models with various prompts to identify and diagnose empty response issues.

3. **Debug Mode**: You can enable debug mode by setting `debug_mode=True` when calling `send_request`, which will log full API responses.
