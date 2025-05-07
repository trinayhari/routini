#!/usr/bin/env python3
"""
Model Comparison Tool

This script compares outputs from different models for the same prompt,
enabling side-by-side comparison of responses, latency, token usage, and cost.
"""

import os
import time
import pandas as pd
import streamlit as st
from typing import Dict, List, Any
from src.api.openrouter_client_enhanced import send_prompt_to_openrouter
from src.config.config_loader import load_config

# Sample comparison prompts that highlight different model capabilities
COMPARISON_PROMPTS = [
    {
        "name": "Factual Knowledge",
        "prompt": "List the 5 largest countries by land area and their capitals.",
        "system": "You are a helpful assistant providing factual information.",
        "description": "Tests factual recall and basic knowledge"
    },
    {
        "name": "Creative Writing",
        "prompt": "Write a short, imaginative story about a time traveler who accidentally changes history.",
        "system": "You are a creative writing assistant with an engaging style.",
        "description": "Tests creative capabilities and storytelling"
    },
    {
        "name": "Coding",
        "prompt": "Write a Python function that checks if a given string is a palindrome. Include docstrings and comments.",
        "system": "You are a programming assistant. Provide clean, correct code.",
        "description": "Tests coding abilities and technical explanation"
    },
    {
        "name": "Reasoning",
        "prompt": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning step by step.",
        "system": "You are a logical assistant who thinks step by step.",
        "description": "Tests logical reasoning and problem-solving"
    },
    {
        "name": "Summarization",
        "prompt": "Summarize the concept of climate change, its causes, effects, and potential solutions in 3-4 sentences.",
        "system": "You are a concise assistant who provides clear summaries.",
        "description": "Tests ability to condense complex topics"
    }
]

# Models to compare (update with models you want to test)
DEFAULT_MODELS = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "openai/gpt-4o",
    "openai/gpt-3.5-turbo",
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct"
]

def get_model_response(model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Get a response from a specific model via OpenRouter"""
    try:
        start_time = time.time()
        response_text, usage_stats, latency = send_prompt_to_openrouter(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=1000
        )
        total_time = time.time() - start_time
        
        # Get cost information from config
        try:
            config = load_config()
            model_info = config.get("models", {}).get(model, {})
            cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
        except Exception:
            cost_per_1k = 0
            
        total_tokens = usage_stats.get("total_tokens", 0)
        estimated_cost = (total_tokens * cost_per_1k) / 1000
        
        return {
            "model": model,
            "success": True,
            "response": response_text,
            "tokens": total_tokens,
            "latency": latency,
            "total_time": total_time,
            "cost": estimated_cost,
            "usage_stats": usage_stats
        }
    except Exception as e:
        return {
            "model": model,
            "success": False,
            "error": str(e),
            "response": f"Error: {str(e)}",
            "tokens": 0,
            "latency": 0,
            "total_time": 0,
            "cost": 0
        }

def streamlit_interface():
    """Streamlit interface for model comparison"""
    st.set_page_config(
        page_title="OpenRouter Chatbot Suite",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("OpenRouter Model Comparison")
    st.markdown("""
    Compare responses from different language models side by side.
    See how they perform on the same prompts across different tasks.
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        st.subheader("Select Models to Compare")
        try:
            config = load_config()
            available_models = list(config.get("models", {}).keys())
        except Exception:
            available_models = DEFAULT_MODELS
            
        selected_models = st.multiselect(
            "Models",
            options=available_models,
            default=available_models[:3] if len(available_models) > 3 else available_models
        )
        
        # Prompt selection
        st.subheader("Select Prompt")
        prompt_options = [p["name"] for p in COMPARISON_PROMPTS]
        selected_prompt_name = st.selectbox("Prompt Type", options=prompt_options)
        
        selected_prompt = next((p for p in COMPARISON_PROMPTS if p["name"] == selected_prompt_name), COMPARISON_PROMPTS[0])
        
        # Custom prompt option
        use_custom_prompt = st.checkbox("Use custom prompt")
        
        custom_system = st.text_area(
            "System message",
            value=selected_prompt["system"],
            disabled=not use_custom_prompt
        )
        
        custom_prompt = st.text_area(
            "User prompt",
            value=selected_prompt["prompt"],
            height=150,
            disabled=not use_custom_prompt
        )
        
        # Compare button
        compare_button = st.button("Compare Models", type="primary")
        
    # Main content area
    if not selected_models:
        st.warning("Please select at least one model to compare.")
    else:
        # Display prompt information
        st.header("Prompt Information")
        
        if use_custom_prompt:
            system_message = custom_system
            user_prompt = custom_prompt
            description = "Custom prompt"
        else:
            system_message = selected_prompt["system"]
            user_prompt = selected_prompt["prompt"]
            description = selected_prompt["description"]
            
        st.markdown(f"**Task Type:** {selected_prompt_name}")
        st.markdown(f"**Description:** {description}")
        st.markdown("**System Message:**")
        st.code(system_message)
        st.markdown("**User Prompt:**")
        st.code(user_prompt)
        
        # Run comparison when button is clicked
        if compare_button:
            st.header("Model Comparison")
            
            # Create messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            # Create progress bar
            progress_text = "Comparing models... This may take a minute."
            progress_bar = st.progress(0, text=progress_text)
            
            # Get responses from each model
            results = []
            
            for i, model in enumerate(selected_models):
                # Update progress
                progress_bar.progress((i / len(selected_models)), 
                                      text=f"Getting response from {model}...")
                
                result = get_model_response(model, messages)
                results.append(result)
                time.sleep(0.5)  # Small delay between API calls
            
            # Complete progress
            progress_bar.progress(1.0, text="Comparison complete!")
            
            # Display results in tabs
            tab_response, tab_metrics, tab_comparison = st.tabs([
                "🗣️ Responses", "📊 Metrics", "📈 Side-by-Side"
            ])
            
            # 1. Response Tab - Show full responses
            with tab_response:
                for result in results:
                    model_name = result["model"].split("/")[-1].upper()
                    provider = result["model"].split("/")[0]
                    
                    with st.expander(f"{model_name} ({provider})"):
                        if result["success"]:
                            st.markdown(result["response"])
                        else:
                            st.error(result["error"])
            
            # 2. Metrics Tab - Show performance metrics
            with tab_metrics:
                # Create metrics dataframe
                metrics_data = []
                
                for result in results:
                    if result["success"]:
                        metrics_data.append({
                            "Model": result["model"].split("/")[-1].upper(),
                            "Provider": result["model"].split("/")[0],
                            "Latency (s)": round(result["latency"], 2),
                            "Total Tokens": result["tokens"],
                            "Estimated Cost ($)": round(result["cost"], 5)
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Bar charts for comparison
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Latency Comparison")
                        latency_chart = pd.DataFrame({
                            "Model": metrics_df["Model"],
                            "Latency (s)": metrics_df["Latency (s)"]
                        }).set_index("Model")
                        st.bar_chart(latency_chart)
                    
                    with col2:
                        st.subheader("Token Usage")
                        tokens_chart = pd.DataFrame({
                            "Model": metrics_df["Model"],
                            "Total Tokens": metrics_df["Total Tokens"]
                        }).set_index("Model")
                        st.bar_chart(tokens_chart)
                    
                    with col3:
                        st.subheader("Cost Comparison")
                        cost_chart = pd.DataFrame({
                            "Model": metrics_df["Model"],
                            "Cost ($)": metrics_df["Estimated Cost ($)"]
                        }).set_index("Model")
                        st.bar_chart(cost_chart)
            
            # 3. Side-by-Side Tab - Show responses side by side
            with tab_comparison:
                # Create columns for each model
                if results:
                    cols = st.columns(len(results))
                    
                    for i, (col, result) in enumerate(zip(cols, results)):
                        model_name = result["model"].split("/")[-1].upper()
                        provider = result["model"].split("/")[0]
                        
                        with col:
                            st.subheader(f"{model_name}")
                            st.caption(f"Provider: {provider}")
                            
                            if result["success"]:
                                st.markdown(result["response"])
                                st.caption(f"Tokens: {result['tokens']} | Latency: {result['latency']:.2f}s | Cost: ${result['cost']:.5f}")
                            else:
                                st.error(result["error"])

def cli_interface():
    """Command-line interface for model comparison"""
    print("\n===== OPENROUTER MODEL COMPARISON =====")
    
    # Select prompt
    print("\nAvailable prompts:")
    for i, prompt in enumerate(COMPARISON_PROMPTS):
        print(f"{i+1}. {prompt['name']}: {prompt['description']}")
    
    prompt_idx = int(input("\nSelect prompt number: ")) - 1
    selected_prompt = COMPARISON_PROMPTS[prompt_idx]
    
    # Select models
    print("\nCommon models:")
    for i, model in enumerate(DEFAULT_MODELS):
        print(f"{i+1}. {model}")
    
    model_input = input("\nEnter model numbers to compare (comma-separated, e.g., '1,3,5'): ")
    model_indices = [int(idx.strip()) - 1 for idx in model_input.split(",")]
    selected_models = [DEFAULT_MODELS[idx] for idx in model_indices if 0 <= idx < len(DEFAULT_MODELS)]
    
    print(f"\n\n----- COMPARING {len(selected_models)} MODELS ON '{selected_prompt['name']}' -----")
    print(f"System: {selected_prompt['system']}")
    print(f"Prompt: {selected_prompt['prompt']}")
    
    # Create messages
    messages = [
        {"role": "system", "content": selected_prompt['system']},
        {"role": "user", "content": selected_prompt['prompt']}
    ]
    
    # Get responses
    results = []
    for model in selected_models:
        print(f"\nGetting response from {model}...")
        result = get_model_response(model, messages)
        results.append(result)
        
        if result["success"]:
            print(f"✓ Success ({result['tokens']} tokens, {result['latency']:.2f}s)")
        else:
            print(f"✗ Error: {result['error']}")
    
    # Show full results
    print("\n\n===== RESULTS =====")
    
    for result in results:
        print(f"\n--- {result['model']} ---")
        
        if result["success"]:
            print(f"Response:\n{result['response'][:500]}...")
            print(f"\nMetrics:")
            print(f"- Tokens: {result['tokens']}")
            print(f"- Latency: {result['latency']:.2f}s")
            print(f"- Estimated cost: ${result['cost']:.5f}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    # Check if running in Streamlit
    if 'STREAMLIT_RUN_PATH' in os.environ:
        streamlit_interface()
    else:
        # Running as standard Python script
        cli_interface() 