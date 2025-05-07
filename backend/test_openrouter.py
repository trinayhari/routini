"""
Test script to debug OpenRouter API integration.
"""
import asyncio
import os
from dotenv import load_dotenv
from app.utils.openrouter import send_request

async def test_openrouter_api():
    """Test OpenRouter API integration"""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OpenRouter API key not found in .env file")
        return
    
    print(f"Using API key: {api_key[:8]}...")
    
    # Simple test prompt
    prompt = "Say hello!"
    
    try:
        # Try to send a request to OpenRouter API
        print("Sending request to OpenRouter API...")
        response_data = await send_request(
            model="openai/gpt-3.5-turbo",  # Using a widely available model
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        print("\nResponse received:")
        
        # Check if we got an error response (dict with 'error' key)
        if isinstance(response_data, dict) and "error" in response_data:
            print(f"ERROR: {response_data.get('message', 'Unknown error')}")
            print(f"Status code: {response_data.get('status_code', 'N/A')}")
        # Check if we got the expected tuple response (text, usage_stats, latency)
        elif isinstance(response_data, tuple) and len(response_data) == 3:
            response_text, usage_stats, latency = response_data
            
            print(f"SUCCESS: Response received in {latency:.2f} seconds")
            print(f"Response text: {response_text}")
            print(f"Tokens: {usage_stats.get('prompt_tokens', 0)} prompt + {usage_stats.get('completion_tokens', 0)} completion")
            print(f"Cost: ${usage_stats.get('cost', 0):.6f}")
        else:
            print(f"Unexpected response format: {type(response_data)}")
            print(f"Response: {response_data}")
    
    except Exception as e:
        print(f"\nError encountered: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_openrouter_api()) 