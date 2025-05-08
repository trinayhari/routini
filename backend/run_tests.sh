#!/bin/bash
# Run error handling tests

# Set environment variable to use mock responses
export USE_MOCK=true

echo "Running OpenRouter error handling tests..."
cd "$(dirname "$0")"
python -m app.test_error_handling

echo ""
echo "Testing client integration..."
python -m app.test_openrouter_client

echo ""
echo "All tests completed" 