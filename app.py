import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client and LangChain LLM
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

# Define prompt templates
feature_prompt = PromptTemplate(
    input_variables=["background", "columns", "target_column"],
    template="Given this background about the dataset: {background}, available columns: {columns}, and the target column to create: {target_column}, generate Python code as plain text to perform feature engineering. Create the target column '{target_column}' in the DataFrame 'df' using the appropriate columns based on the background. Use only the columns provided in the 'columns' list and ensure the code is valid Python syntax (e.g., df['{target_column}'] = df['col1'] - df['col2']). Return only the code without explanations or formatting."
)

forecast_prompt = PromptTemplate(
    input_variables=["task", "data", "context"],
    template="Given this task: {task}, data: {data}, and context: {context}, generate the appropriate code or insight as plain text without markdown, backticks, or additional formatting. For Prophet code, use 'from prophet import Prophet', define 'model' as the Prophet instance, and 'forecast' as the prediction output, ensuring the DataFrame 'df' has 'ds' for dates and 'y' for the target column specified. For insights, provide a detailed analysis of trends, peaks, or dips in the forecast, with actionable business recommendations in a concise paragraph (3-5 sentences), avoiding code or technical jargon, and leveraging the context to tailor the insights."
)

# Create RunnableSequences
feature_chain = RunnableSequence(feature_prompt | llm)
forecast_chain = RunnableSequence(forecast_prompt | llm)

# Load the dataset and return columns
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

# Infer dataset granularity
def infer_granularity(df):
    try:
        df_sorted = df[['ds']].sort_values('ds').drop_duplicates()
        time_diffs = df_sorted['ds'].diff().dropna()
        min_diff = time_diffs.min()
        min_diff_seconds = min_diff.total_seconds()
        if min_diff_seconds <= 86400:  # Daily or less
            return ['D', 'W', 'ME']
        elif min_diff_seconds <= 604800:  # Weekly or less
            return ['W', 'ME']
        else:
            return ['ME']
    except Exception as e:
        st.warning(f"Could not infer granularity: {e}. Defaulting to Weekly.")
        return ['W']

# Aggregate data
def aggregate_data(df, target_column, frequency='W', group_columns=None):
    try:
        df_copy = df.copy()
        if frequency == 'D':
            df_copy['ds'] = df_copy['ds'].dt.floor('D')
        elif frequency == 'W':
            df_copy['ds'] = df_copy['ds'].dt.to_period('W').dt.to_timestamp()
        elif frequency == 'ME':  # Updated from 'M' to 'ME'
            df_copy['ds'] = df_copy['ds'].dt.to_period('M').dt.to_timestamp()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        if group_columns and group_columns[0]:
            groupby_cols = ['ds'] + group_columns
            agg_df = df_copy.groupby(groupby_cols, as_index=False)[target_column].sum()
        else:
            agg_df = df_copy.groupby('ds', as_index=False)[target_column].sum()
        return agg_df
    except Exception as e:
        st.error(f"Error aggregating data: {e}")
        return df

# Feature engineering
def engineer_features(df, target_column, background, columns):
    try:
        if target_column in df.columns:
            st.info(f"Target column '{target_column}' already exists.")
            return df
        
        feature_code = feature_chain.invoke({
            "background": background,
            "columns": ", ".join(columns),
            "target_column": target_column
        }).content
        local_vars = {'df': df.copy()}
        exec(feature_code, globals(), local_vars)
        df = local_vars['df']
        st.success(f"Created '{target_column}'.")
        return df
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        return df

# Forecast for a single group
def run_forecast(df, target_column, periods, frequency, data_color, forecast_color):
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found.")
        return None, None, None, None
    
    df_prophet = df[['ds', target_column]].rename(columns={target_column: 'y'})
    if len(df_prophet.dropna()) < 2:
        st.warning(f"Not enough data for forecasting (less than 2 non-NaN rows).")
        return None, None, None, None
    
    try:
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = model.predict(future)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        last_historical_date = df_prophet['ds'].max()
        historical_data = df_prophet[df_prophet['ds'] <= last_historical_date]
        forecast_data = forecast[forecast['ds'] > last_historical_date]
        
        ax1.plot(historical_data['ds'], historical_data['y'], '-', color=data_color, label='Historical Data')
        ax1.plot(forecast_data['ds'], forecast_data['yhat'], '-', color=forecast_color, label='Forecast')
        ax1.fill_between(forecast_data['ds'], forecast_data['yhat_lower'], forecast_data['yhat_upper'], color=forecast_color, alpha=0.2)
        ax1.legend()
        ax1.set_title(f'{target_column} Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(target_column)
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        fig2 = model.plot_components(forecast, figsize=(10, 8))
        return model, forecast, fig1, fig2
    except Exception as e:
        st.error(f"Error in Prophet model: {e}")
        return None, None, None, None

# Generate insights
def get_insights(forecast, target_column, context):
    try:
        insights = forecast_chain.invoke({
            "task": "Provide a detailed business insights",
            "data": f"forecast for {target_column}: {forecast[['ds', 'yhat']].tail().to_string()}",
            "context": context
        }).content
        return insights
    except Exception as e:
        return f"Error generating insights: {e}"

# Multi-group forecast with descriptive headers and filtering
def run_multi_group_forecast(df, group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=10, filter_group=None, selected_group=None):
    group_title = filter_group if filter_group else " & ".join(group_columns)
    
    # Generate a descriptive header based on group and context
    if filter_group:
        description = f"Analyzing {target_column} trends for top {top_n} {filter_group}s based on {context.lower()}"
    else:
        description = f"Forecasting {target_column} across top {top_n} combinations of {' and '.join(group_columns)} from {context.lower()}"
    st.subheader(description)
    
    agg_df = aggregate_data(df, target_column, frequency, group_columns if not filter_group else [filter_group])
    
    if len(group_columns if not filter_group else [filter_group]) == 1:
        group_values = agg_df[group_columns[0] if not filter_group else filter_group].value_counts().nlargest(top_n).index.tolist()
        combined_groups = [(val,) for val in group_values]
    else:
        group_sums = agg_df.groupby(group_columns)[target_column].sum().nlargest(top_n)
        combined_groups = list(group_sums.index)
    
    fig_compare, ax_compare = plt.subplots(figsize=(12, 6))
    forecasts_dict = {}
    agg_df_dict = {}  # Store aggregated data per group
    
    for i, group_combo in enumerate(combined_groups):
        group_color = plt.cm.tab10(i % 10)  # Unique color per group
        group_data = agg_df.copy()
        
        if len(group_columns if not filter_group else [filter_group]) == 1:
            group_data = group_data[group_data[group_columns[0] if not filter_group else filter_group] == group_combo[0]]
            group_label = str(group_combo[0])
        else:
            for col, val in zip(group_columns, group_combo):
                group_data = group_data[group_data[col] == val]
            group_label = " & ".join([f"{col}={val}" for col, val in zip(group_columns, group_combo)])
        
        if not group_data.empty and len(group_data.dropna()) >= 2:
            try:
                prophet_data = group_data[['ds', target_column]].rename(columns={target_column: 'y'})
                model = Prophet()
                model.fit(prophet_data)
                future = pd.DataFrame({
                    'ds': pd.date_range(start=prophet_data['ds'].max(), periods=periods + 1, freq=frequency)[1:]
                })
                forecast = model.predict(future)
                
                last_date = prophet_data['ds'].max()
                historical_data = prophet_data[prophet_data['ds'] <= last_date]
                forecast_values = forecast[forecast['ds'] > last_date]
                
                # Plot only if selected_group matches or no filter is applied
                if selected_group is None or group_label == selected_group:
                    ax_compare.plot(historical_data['ds'], historical_data['y'], '-', color=group_color, alpha=0.5, label=f"{group_label} (Historical)")
                    ax_compare.plot(forecast_values['ds'], forecast_values['yhat'], '-', color=group_color, label=f"{group_label} (Forecast)")
                
                forecasts_dict[group_label] = forecast
                agg_df_dict[group_label] = group_data  # Store for heatmap and detailed view
            except Exception as e:
                st.warning(f"Could not forecast for {group_label}: {e}")
        else:
            st.warning(f"Skipping {group_label}: Not enough data (less than 2 non-NaN rows).")
    
    ax_compare.set_title(f"{target_column} Forecast Comparison")
    ax_compare.set_xlabel("Date")
    ax_compare.set_ylabel(target_column)
    ax_compare.grid(True, linestyle='--', alpha=0.5)
    ax_compare.legend()
    plt.tight_layout()
    
    if forecasts_dict:
        group_labels = list(forecasts_dict.keys())
        st.session_state[f"heatmap_data_{group_title}"] = (forecasts_dict, agg_df_dict, group_labels, target_column, group_title)
    else:
        st.error("No valid forecasts generated for any group.")
    
    return fig_compare, forecasts_dict, agg_df

# Dynamic Heatmap with Filtering
def create_forecast_heatmap(forecasts_dict=None, group_labels=None, target_column=None, group_title=None, agg_df_dict=None):
    if not all([forecasts_dict, group_labels, target_column, group_title]):
        st.warning("Missing data for heatmap generation.")
        return None
    
    st.subheader(f"Forecast Heatmap by {group_title}")
    
    selected_groups = st.multiselect(
        "Filter Groups for Heatmap",
        options=group_labels,
        default=group_labels[:min(10, len(group_labels))],
        key=f"heatmap_filter_{group_title}"
    )
    
    if not selected_groups:
        st.warning("Please select at least one group to display the heatmap.")
        return None
    
    all_forecasts = pd.DataFrame()
    for group_label in selected_groups:
        if group_label in forecasts_dict:
            forecast = forecasts_dict[group_label]
            if agg_df_dict and group_label in agg_df_dict:
                last_historical_date = agg_df_dict[group_label]['ds'].max()
            else:
                last_historical_date = forecast['ds'][forecast['yhat_upper'].isna()].max() or forecast['ds'].iloc[len(forecast)//2]
            forecast_future = forecast[forecast['ds'] > last_historical_date].copy()
            if not forecast_future.empty:
                forecast_future['group'] = group_label
                all_forecasts = pd.concat([all_forecasts, forecast_future[['ds', 'yhat', 'group']]])
            else:
                st.warning(f"No future data for group {group_label} after {last_historical_date}")
    
    if all_forecasts.empty:
        st.warning("No future forecast data available for the selected groups.")
        return None
    
    pivot_df = all_forecasts.pivot(index='group', columns='ds', values='yhat')
    pivot_df.columns = pivot_df.columns.strftime('%Y-%m-%d')
    
    fig, ax = plt.subplots(figsize=(14, len(selected_groups) * 0.5 + 2))
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5, ax=ax)
    ax.set_title(f"{target_column} Forecast Heatmap by {group_title}")
    ax.set_ylabel("Group")
    ax.set_xlabel("Date")
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.download_button(
        label="Download Heatmap Data",
        data=pivot_df.reset_index().to_csv(index=False),
        file_name=f"forecast_heatmap_{target_column}_by_{group_title}.csv",
        mime="text/csv",
    )
    return fig

# Convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False)

# Initialize session state
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

# Streamlit UI
st.title("Time Series Forecasting - Generative AI")
st.markdown("Upload your dataset, specify the target column and its calculation in the context, and customize the forecast.")

with st.sidebar:
    st.header("Options")
    context = st.text_area(
        "Dataset context (e.g., 'Retail sales data, Weekly_Sales is the target')",
        value="This dataset tracks retail sales, Weekly_Sales is the target."
    )
    date_column = st.text_input("Date column name", value="Date")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    df, columns = load_data(file=uploaded_file, date_column=date_column) if uploaded_file else (None, [])
    if df is not None:
        st.write("Columns:", ", ".join(columns))
        st.write(f"Raw data date range: {df['ds'].min()} to {df['ds'].max()}")
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

        st.session_state.forecast_results = {
            'df': df,
            'target_column': target_column,
            'periods': periods,
            'frequency': frequency,
            'data_color': data_color,
            'forecast_color': forecast_color,
            'selected_group_columns': selected_group_columns,
            'top_n': top_n if 'top_n' in locals() else 10,
            'context': context
        }
        
        all_agg_df = aggregate_data(df, target_column, frequency)
        all_model, all_forecast, all_fig1, all_fig2 = run_forecast(all_agg_df, target_column, periods, frequency, data_color, forecast_color)
        st.session_state.forecast_results['all'] = (all_model, all_forecast, all_fig1, all_fig2, all_agg_df)
        
        if enable_groupby and selected_group_columns:
            combined_fig, combined_forecasts, combined_agg_df = run_multi_group_forecast(
                df, selected_group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=top_n
            )
            st.session_state.forecast_results['combined'] = (combined_fig, combined_forecasts, combined_agg_df)
            
            if len(selected_group_columns) >= 1:
                primary_fig, primary_forecasts, primary_agg_df = run_multi_group_forecast(
                    df, selected_group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=top_n, filter_group=selected_group_columns[0]
                )
                st.session_state.forecast_results['primary'] = (primary_fig, primary_forecasts, primary_agg_df)
            
            if len(selected_group_columns) >= 2:
                secondary_fig, secondary_forecasts, secondary_agg_df = run_multi_group_forecast(
                    df, selected_group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=top_n, filter_group=selected_group_columns[1]
                )
                st.session_state.forecast_results['secondary'] = (secondary_fig, secondary_forecasts, secondary_agg_df)

# Filter selection and display
if 'forecast_results' in st.session_state and st.session_state.forecast_results:
    results = st.session_state.forecast_results
    df = results['df']
    target_column = results['target_column']
    periods = results['periods']
    frequency = results['frequency']
    data_color = results['data_color']
    forecast_color = results['forecast_color']
    selected_group_columns = results['selected_group_columns']
    top_n = results['top_n']
    context = results['context']

    filter_options = ["All Data Combined"]
    if selected_group_columns:
        filter_options.append(f"Combined ({' & '.join(selected_group_columns)})")
        filter_options.append(selected_group_columns[0])
        if len(selected_group_columns) >= 2:
            filter_options.append(selected_group_columns[1])
    
    st.subheader("Filter Forecast Results")
    selected_filter = st.selectbox("Choose a filter to view results:", filter_options, key="filter_select")

    if selected_filter == "All Data Combined" and 'all' in results:
        all_model, all_forecast, all_fig1, all_fig2, all_agg_df = results['all']
        if all_model:
            st.subheader("Forecast Results (All Data Combined)")
            st.write(all_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            st.write("Insights:", get_insights(all_forecast, target_column, context))
            if all_fig1:
                st.pyplot(all_fig1)
            else:
                st.warning("No forecast plot generated for All Data Combined.")
            if all_fig2:
                st.pyplot(all_fig2)
            else:
                st.warning("No components plot generated for All Data Combined.")
            st.download_button(
                label="Download Full Dataset",
                data=convert_df_to_csv(df),
                file_name="dataset_with_features.csv",
                mime="text/csv",
            )
    
    elif selected_group_columns:
        if selected_filter == f"Combined ({' & '.join(selected_group_columns)})" and 'combined' in results:
            combined_fig, combined_forecasts, combined_agg_df = results['combined']
            st.subheader(f"Forecast Results by {' & '.join(selected_group_columns)}")
            st.write(f"Aggregated data ({frequency} frequency):")
            st.dataframe(combined_agg_df.head())
            if combined_fig:
                st.pyplot(combined_fig)
            else:
                st.warning("No comparison plot generated for Combined.")
            if combined_forecasts:
                group_title = " & ".join(selected_group_columns)
                if f"heatmap_data_{group_title}" in st.session_state:
                    forecasts_dict, agg_df_dict, group_labels, target_col, title = st.session_state[f"heatmap_data_{group_title}"]
                    heatmap_fig = create_forecast_heatmap(forecasts_dict, group_labels, target_col, title, agg_df_dict)
                    if not heatmap_fig:
                        st.warning("Heatmap generation failed.")
                else:
                    st.warning(f"No heatmap data found for {group_title}.")
                all_forecasts = pd.concat([forecast[['ds', 'yhat']].assign(group=group) for group, forecast in combined_forecasts.items()])
                st.download_button(
                    label="Download Combined Forecasts",
                    data=convert_df_to_csv(all_forecasts),
                    file_name=f"combined_forecasts_{target_column}.csv",
                    mime="text/csv",
                )
                detailed_group = st.selectbox("Select a group for detailed view", options=list(combined_forecasts.keys()), key="combined_detail")
                if detailed_group:
                    st.subheader(f"Detailed Forecast for {detailed_group}")
                    forecast = combined_forecasts[detailed_group]
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                    st.write("Insights:", get_insights(forecast, target_column, context))
                    group_data = combined_agg_df
                    parts = detailed_group.split(' & ')
                    store_val = parts[0].split('=')[1]
                    dept_val = parts[1].split('=')[1] if len(parts) > 1 else None
                    group_data = group_data[group_data[selected_group_columns[0]] == store_val]
                    if dept_val and len(selected_group_columns) > 1:
                        group_data = group_data[group_data[selected_group_columns[1]] == dept_val]
                    st.write(f"Detailed view for {detailed_group} has {len(group_data)} rows")
                    st.write(f"Group data sample:\n{group_data.head()}")
                    model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                    if fig1:
                        st.pyplot(fig1)
                    if fig2:
                        st.pyplot(fig2)
        
        elif selected_filter == selected_group_columns[0] and 'primary' in results:
            primary_fig, primary_forecasts, primary_agg_df = results['primary']
            st.subheader(f"Forecast Results by {selected_group_columns[0]}")
            st.write(f"Aggregated data ({frequency} frequency):")
            st.dataframe(primary_agg_df.head())
            if primary_fig:
                st.pyplot(primary_fig)
            else:
                st.warning("No comparison plot generated for Primary.")
            if primary_forecasts:
                group_title = selected_group_columns[0]
                group_labels = list(primary_forecasts.keys())
                selected_group = st.selectbox(
                    f"Select {selected_group_columns[0]} to view",
                    options=["All"] + group_labels,
                    key=f"primary_group_select"
                )
                if f"heatmap_data_{group_title}" in st.session_state:
                    forecasts_dict, agg_df_dict, all_labels, target_col, title = st.session_state[f"heatmap_data_{group_title}"]
                    heatmap_fig = create_forecast_heatmap(forecasts_dict, all_labels, target_col, title, agg_df_dict)
                    if not heatmap_fig:
                        st.warning("Heatmap generation failed.")
                else:
                    st.warning(f"No heatmap data found for {group_title}.")
                all_forecasts = pd.concat([forecast[['ds', 'yhat']].assign(group=group) for group, forecast in primary_forecasts.items()])
                st.download_button(
                    label=f"Download {selected_group_columns[0]} Forecasts",
                    data=convert_df_to_csv(all_forecasts),
                    file_name=f"{selected_group_columns[0]}_forecasts_{target_column}.csv",
                    mime="text/csv",
                )
                if selected_group != "All":
                    st.subheader(f"Detailed Forecast for {selected_group}")
                    forecast = primary_forecasts[selected_group]
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                    st.write("Insights:", get_insights(forecast, target_column, context))
                    group_data = primary_agg_df
                    parts = selected_group.split(' & ')
                    store_val = parts[0].split('=')[1]
                    dept_val = parts[1].split('=')[1] if len(parts) > 1 else None
                    group_data = group_data[group_data[selected_group_columns[0]] == store_val]
                    if dept_val and len(selected_group_columns) > 1:
                        group_data = group_data[group_data[selected_group_columns[1]] == dept_val]
                    st.write(f"Detailed view for {selected_group} has {len(group_data)} rows")
                    st.write(f"Group data sample:\n{group_data.head()}")
                    model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                    if fig1:
                        st.pyplot(fig1)
                    if fig2:
                        st.pyplot(fig2)
        
        elif len(selected_group_columns) >= 2 and selected_filter == selected_group_columns[1] and 'secondary' in results:
            secondary_fig, secondary_forecasts, secondary_agg_df = results['secondary']
            st.subheader(f"Forecast Results by {selected_group_columns[1]}")
            st.write(f"Aggregated data ({frequency} frequency):")
            st.dataframe(secondary_agg_df.head())
            if secondary_fig:
                st.pyplot(secondary_fig)
            else:
                st.warning("No comparison plot generated for Secondary.")
            if secondary_forecasts:
                group_title = selected_group_columns[1]
                group_labels = list(secondary_forecasts.keys())
                selected_group = st.selectbox(
                    f"Select {selected_group_columns[1]} to view",
                    options=["All"] + group_labels,
                    key=f"secondary_group_select"
                )
                if f"heatmap_data_{group_title}" in st.session_state:
                    forecasts_dict, agg_df_dict, all_labels, target_col, title = st.session_state[f"heatmap_data_{group_title}"]
                    heatmap_fig = create_forecast_heatmap(forecasts_dict, all_labels, target_col, title, agg_df_dict)
                    if not heatmap_fig:
                        st.warning("Heatmap generation failed.")
                else:
                    st.warning(f"No heatmap data found for {group_title}.")
                all_forecasts = pd.concat([forecast[['ds', 'yhat']].assign(group=group) for group, forecast in secondary_forecasts.items()])
                st.download_button(
                    label=f"Download {selected_group_columns[1]} Forecasts",
                    data=convert_df_to_csv(all_forecasts),
                    file_name=f"{selected_group_columns[1]}_forecasts_{target_column}.csv",
                    mime="text/csv",
                )
                if selected_group != "All":
                    st.subheader(f"Detailed Forecast for {selected_group}")
                    forecast = secondary_forecasts[selected_group]
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                    st.write("Insights:", get_insights(forecast, target_column, context))
                    group_data = secondary_agg_df
                    parts = selected_group.split(' & ')
                    store_val = parts[0].split('=')[1]
                    dept_val = parts[1].split('=')[1] if len(parts) > 1 else None
                    group_data = group_data[group_data[selected_group_columns[0]] == store_val]
                    if dept_val and len(selected_group_columns) > 1:
                        group_data = group_data[group_data[selected_group_columns[1]] == dept_val]
                    st.write(f"Detailed view for {selected_group} has {len(group_data)} rows")
                    st.write(f"Group data sample:\n{group_data.head()}")
                    model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                    if fig1:
                        st.pyplot(fig1)
                    if fig2:
                        st.pyplot(fig2)
