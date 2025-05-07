#!/usr/bin/env python3
"""
Cost Dashboard

A Streamlit dashboard for visualizing OpenRouter API usage costs.
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from datetime import datetime, timedelta
from src.utils.cost_tracker import CostTracker
from streamlit_echarts import st_echarts

# Set page configuration
st.set_page_config(
    page_title="OpenRouter Cost Dashboard",
    page_icon="💰",
    layout="wide"
)

def load_config():
    """Load the configuration file"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {"models": {}}

def format_cost(cost):
    """Format cost as a currency string"""
    return f"${cost:.5f}"

def format_tokens(tokens):
    """Format token count with commas"""
    return f"{int(tokens):,}"

def main():
    # Load configuration
    config = load_config()
    
    # Initialize cost tracker
    cost_tracker = CostTracker(config)
    
    # Header
    st.title("📊 OpenRouter API Cost Dashboard")
    st.write("Track and analyze your API usage and costs")
    
    # Main content
    tabs = st.tabs(["Overview", "Daily Breakdown", "Model Comparison", "Raw Data"])
    
    # === Overview Tab ===
    with tabs[0]:
        # Get summaries
        today_summary = cost_tracker.get_daily_summary()
        session_summary = cost_tracker.get_session_summary()
        
        # Create cost data
        cost_trends = cost_tracker.get_cost_trends(days=30)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Today's Cost", 
                value=format_cost(today_summary["total_cost"]),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Today's Tokens", 
                value=format_tokens(today_summary["total_tokens"]),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Session Cost", 
                value=format_cost(session_summary["total_cost"]),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Session Requests", 
                value=str(session_summary["calls"]),
                delta=None
            )
        
        # Cost trend chart
        st.subheader("Cost Trends")
        if not cost_trends.empty:
            # Convert date to string for chart
            cost_trends['date_str'] = cost_trends['date'].astype(str)
            
            # Prepare data for ECharts
            option = {
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["Cost", "Tokens"]},
                "xAxis": [{
                    "type": "category",
                    "data": cost_trends['date_str'].tolist()
                }],
                "yAxis": [
                    {
                        "type": "value",
                        "name": "Cost ($)",
                        "position": "left",
                    },
                    {
                        "type": "value",
                        "name": "Tokens",
                        "position": "right",
                    }
                ],
                "series": [
                    {
                        "name": "Cost",
                        "type": "line",
                        "data": cost_trends['cost'].tolist(),
                        "yAxisIndex": 0,
                        "color": "#f63366"
                    },
                    {
                        "name": "Tokens",
                        "type": "bar",
                        "data": cost_trends['total_tokens'].tolist(),
                        "yAxisIndex": 1,
                        "color": "#4169E1"
                    }
                ]
            }
            
            # Render chart
            st_echarts(option, height="400px")
        else:
            st.info("No cost data available yet. Start using the API to see cost trends.")
        
        # Model usage
        st.subheader("Model Usage Breakdown")
        session_models = session_summary.get("models", {})
        
        if session_models:
            # Prepare data for pie chart
            model_names = []
            model_costs = []
            model_tokens = []
            
            for model, stats in session_models.items():
                model_names.append(model.split("/")[-1])  # Just the model name, not provider
                model_costs.append(stats["cost"])
                model_tokens.append(stats["tokens"])
            
            # Pie chart options
            pie_option = {
                "tooltip": {"trigger": "item"},
                "legend": {"top": "5%", "left": "center"},
                "series": [
                    {
                        "name": "Model Cost",
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": False,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "#fff",
                            "borderWidth": 2
                        },
                        "label": {
                            "show": False,
                            "position": "center"
                        },
                        "emphasis": {
                            "label": {
                                "show": True,
                                "fontSize": "16",
                                "fontWeight": "bold"
                            }
                        },
                        "labelLine": {
                            "show": False
                        },
                        "data": [
                            {"value": cost, "name": name} 
                            for name, cost in zip(model_names, model_costs)
                        ]
                    }
                ]
            }
            
            # Create columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cost by Model")
                st_echarts(pie_option, height="400px")
            
            with col2:
                # Create token usage chart
                token_option = {
                    "tooltip": {"trigger": "item"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Token Usage",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2
                            },
                            "label": {
                                "show": False,
                                "position": "center"
                            },
                            "emphasis": {
                                "label": {
                                    "show": True,
                                    "fontSize": "16",
                                    "fontWeight": "bold"
                                }
                            },
                            "labelLine": {
                                "show": False
                            },
                            "data": [
                                {"value": tokens, "name": name} 
                                for name, tokens in zip(model_names, model_tokens)
                            ]
                        }
                    ]
                }
                
                st.subheader("Token Usage by Model")
                st_echarts(token_option, height="400px")
        else:
            st.info("No model usage data available yet.")
    
    # === Daily Breakdown Tab ===
    with tabs[1]:
        st.header("Daily Cost Analysis")
        
        # Date selector
        date = st.date_input(
            "Select date", 
            value=datetime.now().date(),
            min_value=(datetime.now() - timedelta(days=90)).date(),
            max_value=datetime.now().date()
        )
        
        # Get daily summary
        date_str = date.strftime("%Y-%m-%d")
        daily_summary = cost_tracker.get_daily_summary(date_str)
        
        st.subheader(f"Summary for {date_str}")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cost", format_cost(daily_summary["total_cost"]))
        
        with col2:
            st.metric("Total Tokens", format_tokens(daily_summary["total_tokens"]))
        
        with col3:
            st.metric("API Calls", str(daily_summary["calls"]))
        
        # Show model breakdown
        st.subheader("Model Breakdown")
        
        model_stats = daily_summary.get("models", {})
        if model_stats:
            model_data = []
            
            for model, stats in model_stats.items():
                model_data.append({
                    "Model": model,
                    "Cost": format_cost(stats["cost"]),
                    "Tokens": format_tokens(stats["tokens"]),
                    "Calls": stats["calls"],
                    "Avg Tokens per Call": int(stats["tokens"] / stats["calls"]) if stats["calls"] > 0 else 0,
                    "Avg Cost per Call": format_cost(stats["cost"] / stats["calls"]) if stats["calls"] > 0 else "$0.00000"
                })
            
            # Display as table
            if model_data:
                st.dataframe(pd.DataFrame(model_data), use_container_width=True)
            else:
                st.info(f"No data available for {date_str}")
        else:
            st.info(f"No data available for {date_str}")
    
    # === Model Comparison Tab ===
    with tabs[2]:
        st.header("Model Cost Comparison")
        
        # Get model info from config
        models = config.get("models", {})
        
        if models:
            # Create model comparison table
            comparison_data = []
            
            for model_id, model_info in models.items():
                comparison_data.append({
                    "Model": model_id,
                    "Name": model_info.get("name", model_id),
                    "Provider": model_info.get("provider", "Unknown"),
                    "Cost per 1K Tokens": format_cost(model_info.get("cost_per_1k_tokens", 0)),
                    "Max Tokens": format_tokens(model_info.get("max_tokens", 0))
                })
            
            # Display as table
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Cost estimation tool
            st.subheader("Cost Estimation Tool")
            
            # Select model
            selected_model = st.selectbox(
                "Select Model",
                options=list(models.keys()),
                format_func=lambda x: f"{models[x].get('name', x)} ({x})"
            )
            
            # Token input
            tokens = st.number_input("Number of tokens", min_value=1, value=1000, step=100)
            
            # Calculate cost
            if selected_model and tokens:
                model_info = models.get(selected_model, {})
                cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
                estimated_cost = (tokens * cost_per_1k) / 1000
                
                st.success(f"Estimated cost for {tokens:,} tokens with {model_info.get('name', selected_model)}: {format_cost(estimated_cost)}")
            
            # Cost comparison chart
            st.subheader("Cost Comparison Chart")
            
            # Prepare data for comparison chart
            token_values = [1000, 10000, 100000, 1000000]
            chart_data = []
            
            for token_count in token_values:
                row = {"Tokens": f"{token_count:,}"}
                
                for model_id, model_info in models.items():
                    model_name = model_info.get("name", model_id)
                    cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
                    row[model_name] = (token_count * cost_per_1k) / 1000
                
                chart_data.append(row)
            
            # Convert to DataFrame
            chart_df = pd.DataFrame(chart_data)
            
            # Melt DataFrame for better plotting
            melted_df = pd.melt(
                chart_df, 
                id_vars=["Tokens"], 
                var_name="Model", 
                value_name="Cost"
            )
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model in melted_df["Model"].unique():
                model_data = melted_df[melted_df["Model"] == model]
                ax.plot(model_data["Tokens"], model_data["Cost"], marker='o', label=model)
            
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Cost ($)")
            ax.set_title("Cost Comparison by Token Count")
            ax.legend()
            ax.grid(True)
            
            # Display the plot
            st.pyplot(fig)
            
            # Logarithmic scale version
            st.subheader("Cost Comparison (Log Scale)")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model in melted_df["Model"].unique():
                model_data = melted_df[melted_df["Model"] == model]
                ax.plot(model_data["Tokens"], model_data["Cost"], marker='o', label=model)
            
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Cost ($)")
            ax.set_title("Cost Comparison by Token Count (Log Scale)")
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)
            
            # Display the plot
            st.pyplot(fig)
            
        else:
            st.info("No model information available in configuration.")
    
    # === Raw Data Tab ===
    with tabs[3]:
        st.header("Raw Usage Data")
        
        # Show raw cost data
        if not cost_tracker.cost_data.empty:
            st.dataframe(cost_tracker.cost_data.sort_values("timestamp", ascending=False), use_container_width=True)
            
            # Download button
            csv = cost_tracker.cost_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="openrouter_cost_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No usage data available yet.")
    
    # Footer
    st.markdown("---")
    st.caption("OpenRouter Cost Dashboard | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main() 