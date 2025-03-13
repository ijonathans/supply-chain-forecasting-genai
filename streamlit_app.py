import streamlit as st
import os
import sys
import traceback

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OpenAI API key not found. Please set it in the Streamlit Cloud dashboard under 'Secrets'.")
    st.info("To set up your API key in Streamlit Cloud:")
    st.code("""
1. Go to your app dashboard
2. Click on "Settings" ⚙️ 
3. Go to "Secrets" section
4. Add a new secret with key: OPENAI_API_KEY and your API key as the value
    """)
    api_key = st.text_input("Or enter your OpenAI API key here (not recommended for production):", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set! The app will now load.")
    else:
        st.stop()

try:
    # Redirect to the main app
    st.write("Loading application...")
    from app import *
    st.write("Application loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading the application: {str(e)}")
    st.code(traceback.format_exc())
    st.info("Please check the logs for more details and ensure all dependencies are installed correctly.")

# This file serves as an entry point for Streamlit Cloud
# It will automatically use the app.py file for the main functionality
