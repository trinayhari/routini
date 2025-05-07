import streamlit as st
import yaml

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}

def initialize_router(config):
    """Initialize a router (placeholder)"""
    return None  # Replace with actual router logic if needed

def initialize_session_state():
    """Initialize session state for the chatbot"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = []
    if "conversation_cost" not in st.session_state:
        st.session_state.conversation_cost = 0.0
    if "manual_model_selection" not in st.session_state:
        st.session_state.manual_model_selection = False
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = True

def display_chat_messages():
    """Display chat messages in the UI"""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

def process_user_input(router):
    """Process user input and generate a response (placeholder logic)"""
    user_input = st.text_input("Type your message:", key="chat_input")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Placeholder bot response
        bot_response = f"Echo: {user_input}"
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

def display_cost_summary():
    """Display a placeholder cost summary"""
    st.info(f"Total conversation cost: ${st.session_state.conversation_cost:.4f}") 