#!/usr/bin/env python3
"""
Check OpenRouter API Key

Utility to check if the OpenRouter API key is properly set in the environment.
"""

import os
import sys

def check_api_key():
    """Check if the OpenRouter API key is properly set"""
    print("Checking for OpenRouter API key...")
    
    # Check if the environment variable exists
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if api_key is None:
        print("\n❌ ERROR: OPENROUTER_API_KEY environment variable is not set")
        print("\nTo set it, run one of the following commands depending on your shell:")
        print("\nBash/Zsh (macOS/Linux):")
        print("  export OPENROUTER_API_KEY=your_api_key_here")
        print("\nWindows Command Prompt:")
        print("  set OPENROUTER_API_KEY=your_api_key_here")
        print("\nWindows PowerShell:")
        print("  $env:OPENROUTER_API_KEY = 'your_api_key_here'")
        print("\nOr create a .env file in the project root with:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        return False
    
    # Check if the API key is empty
    if api_key.strip() == "":
        print("\n❌ ERROR: OPENROUTER_API_KEY environment variable is set but empty")
        return False
    
    # Check for common formatting issues
    if api_key.startswith('"') and api_key.endswith('"'):
        print("\n⚠️ WARNING: Your API key has double quotes around it")
        print("This may cause issues. Remove the quotes from the environment variable.")
        api_key = api_key.strip('"')
    
    if api_key.startswith("'") and api_key.endswith("'"):
        print("\n⚠️ WARNING: Your API key has single quotes around it")
        print("This may cause issues. Remove the quotes from the environment variable.")
        api_key = api_key.strip("'")
    
    if len(api_key) < 20:
        print("\n⚠️ WARNING: Your API key appears to be too short for a typical API key")
        print("Please verify that you're using the correct key.")
    
    # Mask most of the API key for security
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***masked***"
    print(f"\n✅ OPENROUTER_API_KEY is set: {masked_key}")
    
    # Check for common issues with API key format
    if not api_key.startswith("sk-"):
        print("\n⚠️ WARNING: Your OpenRouter API key doesn't start with 'sk-'")
        print("This may not be an issue, but most OpenRouter API keys begin with 'sk-'")
    
    return True

def check_env_file():
    """Check if a .env file exists and contains the API key"""
    if os.path.exists(".env"):
        print("\nFound .env file in the current directory")
        with open(".env", "r") as f:
            env_contents = f.read()
            if "OPENROUTER_API_KEY" in env_contents:
                print("✅ OPENROUTER_API_KEY is defined in .env file")
                
                # Try to extract the key to check for formatting issues
                import re
                match = re.search(r'OPENROUTER_API_KEY=(.+)', env_contents)
                if match:
                    key = match.group(1).strip()
                    if key.startswith('"') and key.endswith('"'):
                        print("⚠️ API key in .env has double quotes which may cause issues")
                    if key.startswith("'") and key.endswith("'"):
                        print("⚠️ API key in .env has single quotes which may cause issues")
            else:
                print("❌ OPENROUTER_API_KEY is not defined in .env file")
    else:
        print("\n.env file not found in the current directory")
        print("Consider creating one with your API key")

def check_dotenv_package():
    """Check if python-dotenv is installed and working"""
    try:
        import dotenv
        print("\n✅ python-dotenv package is installed")
        
        # Check if it's loaded
        if hasattr(dotenv, 'load_dotenv'):
            dotenv.load_dotenv()
            print("✅ Attempted to load environment variables from .env file")
            
            # Re-check the API key after loading .env
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                print("✅ API key found after loading .env")
            else:
                print("❌ API key still not found after loading .env")
    except ImportError:
        print("\n❌ python-dotenv package is not installed")
        print("Install it with: pip install python-dotenv")

def main():
    """Main function to check API key settings"""
    print("=== OpenRouter API Key Checker ===")
    check_api_key()
    check_env_file()
    check_dotenv_package()
    
    print("\n=== Testing Direct API Key Access ===")
    print("Enter your OpenRouter API key to test (will not be stored):")
    api_key = input("API Key: ").strip()
    
    if api_key:
        print("\nTesting direct API access with provided key...")
        try:
            import requests
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://api-key-checker.test"
            }
            
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"\n✅ SUCCESS: API key works! Received list of {len(response.json()['data'])} models")
                print("Your API key is valid and working correctly")
            else:
                print(f"\n❌ ERROR: Received status code {response.status_code}")
                print(f"Response: {response.text[:200]}")
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
    else:
        print("\nNo API key provided for testing")
    
    print("\n=== Recommendations ===")
    print("1. Ensure you're using the correct API key from your OpenRouter account")
    print("2. Set the API key without quotes:")
    print("   export OPENROUTER_API_KEY=sk-your-key-here")
    print("3. For .env file usage, make sure you're loading it with python-dotenv")
    print("4. Try passing the API key directly to the function:")
    print("   send_prompt_to_openrouter(..., api_key='your-key-here')")

if __name__ == "__main__":
    main() 