import streamlit as st
import os
import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client and LangChain LLM
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in .env.")
    st.stop()

client = OpenAI(api_key=api_key)
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

# Your existing functions (unchanged for brevity)
def load_data(file=None, date_column='ds', filename='time_series_data.csv'):
    try:
        if file:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(filename)
        
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        
        df = df.rename(columns={date_column: 'ds'})
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        if df['ds'].isna().any():
            raise ValueError("Some dates could not be parsed.")
        
        return df, df.columns.tolist()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, []

# ... (include all your other functions: infer_granularity, aggregate_data, etc.)

# Initialize session state
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

# Streamlit UI
st.set_page_config(page_title="Time Series Forecasting AI", layout="wide")
st.title("Time Series Forecasting - Generative AI")
st.markdown("Welcome! Upload a CSV file with time series data to generate forecasts. The dataset should include a date column and relevant metrics.")

with st.sidebar:
    st.header("Options")
    context = st.text_area(
        "Dataset context (e.g., 'Retail sales data, Weekly_Sales is the target')",
        value="This dataset tracks retail sales, Weekly_Sales is the target."
    )
    date_column = st.text_input("Date column name", value="Date")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    filename = 'time_series_data.csv'
    if uploaded_file:
        df, columns = load_data(file=uploaded_file, date_column=date_column)
    elif os.path.exists(filename):
        df, columns = load_data(filename=filename, date_column=date_column)
    else:
        df, columns = None, []
        st.info("No default data file (time_series_data.csv) found in the repository. Please upload a CSV file to proceed.")

    if df is not None:
        st.write("Columns detected:", ", ".join(columns))
        st.write(f"Data date range: {df['ds'].min()} to {df['ds'].max()}")
        target_column = st.text_input("Target column to forecast", value="Weekly_Sales")
        
        enable_groupby = st.checkbox("Enable Group By", value=True)
        selected_group_columns = []
        
        if enable_groupby:
            non_date_columns = [col for col in df.columns if col != 'ds']
            group_col1 = st.selectbox("Primary Group Column", options=[""] + non_date_columns, index=non_date_columns.index('Store') + 1 if 'Store' in non_date_columns else 0, key="group1")
            if group_col1:
                selected_group_columns.append(group_col1)
                group_col2 = st.selectbox("Secondary Group Column (optional)", options=[""] + [col for col in non_date_columns if col != group_col1], index=non_date_columns.index('Dept') + 1 if 'Dept' in non_date_columns and 'Dept' != group_col1 else 0, key="group2")
                if group_col2:
                    selected_group_columns.append(group_col2)
            
            if selected_group_columns:
                top_n = st.slider("Number of Top Groups to Compare", min_value=2, max_value=20, value=10)
        
        granularity_options = infer_granularity(df)
        frequency_map = {'D': 'Daily', 'W': 'Weekly', 'ME': 'Monthly'}
        frequency = st.selectbox("Forecast Frequency", options=[frequency_map[f] for f in granularity_options], index=1 if 'W' in granularity_options else 0)
        frequency = [k for k, v in frequency_map.items() if v == frequency][0]
        
        period_label = f"Forecast Periods ({frequency_map[frequency].lower()[:-2]}s)"
        periods = st.slider(period_label, min_value=1, max_value=52 if frequency == 'W' else 12 if frequency == 'ME' else 365, value=12)
        
        data_color = st.color_picker("Historical Data Color", value="#000000")
        forecast_color = st.color_picker("Forecast Color", value="#FF0000")
    else:
        target_column = "Weekly_Sales"
        frequency = 'W'
        periods = 12
        top_n = 10
        data_color = '#000000'
        forecast_color = '#FF0000'
        enable_groupby = False
        selected_group_columns = []
    
    run_button = st.button("Generate Forecast", disabled=df is None)

if run_button and df is not None:
    with st.spinner("Generating forecasts..."):
        df = engineer_features(df, target_column, context, columns)
        st.write("Columns after feature engineering:", ", ".join(df.columns))
        # ... (rest of your forecasting logic unchanged)