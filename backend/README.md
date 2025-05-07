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
