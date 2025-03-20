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
import traceback
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully.")
except Exception as e:
    st.error(f"Error loading environment variables: {str(e)}")
    logger.error(f"Error loading environment variables: {str(e)}")

# Initialize OpenAI client and LangChain LLM
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("API key not found in environment variables.")
        logger.warning("API key not found in environment variables.")
    
    client = OpenAI(api_key=api_key)
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)
    logger.info("OpenAI client and LangChain LLM initialized successfully.")
except Exception as e:
    st.error(f"Error initializing OpenAI client or LangChain LLM: {str(e)}")
    logger.error(f"Error initializing OpenAI client or LangChain LLM: {str(e)}")
    st.code(traceback.format_exc())

# Define prompt templates
try:
    feature_prompt = PromptTemplate(
        input_variables=["background", "columns", "target_column"],
        template="Given this background about the dataset: {background}, available columns: {columns}, and the target column to create: {target_column}, generate Python code as plain text to perform feature engineering. Create the target column '{target_column}' in the DataFrame 'df' using the appropriate columns based on the background. Use only the columns provided in the 'columns' list and ensure the code is valid Python syntax (e.g., df['{target_column}'] = df['col1'] - df['col2']). Return only the code without explanations or formatting."
    )

    forecast_prompt = PromptTemplate(
        input_variables=["task", "data", "context"],
        template="Given this task: {task}, data: {data}, and context: {context}, generate the appropriate code or insight as plain text without markdown, backticks, or additional formatting. For Prophet code, use 'from prophet import Prophet', define 'model' as the Prophet instance, and 'forecast' as the prediction output, ensuring the DataFrame 'df' has 'ds' for dates and 'y' for the target column specified. For insights, provide a detailed analysis of trends, peaks, or dips in the forecast, with actionable business recommendations in a concise paragraph (3-5 sentences), avoiding code or technical jargon, and leveraging the context to tailor the insights."
    )
    logger.info("Prompt templates defined successfully.")
except Exception as e:
    st.error(f"Error defining prompt templates: {str(e)}")
    logger.error(f"Error defining prompt templates: {str(e)}")
    st.code(traceback.format_exc())

# Create RunnableSequences
try:
    feature_chain = RunnableSequence(feature_prompt | llm)
    forecast_chain = RunnableSequence(forecast_prompt | llm)
    logger.info("RunnableSequences created successfully.")
except Exception as e:
    st.error(f"Error creating RunnableSequences: {str(e)}")
    logger.error(f"Error creating RunnableSequences: {str(e)}")
    st.code(traceback.format_exc())

# Load the dataset and return columns
def load_data(file=None, date_column='ds', filename='time_series_data.csv'):
    logger.info(f"Loading data with date_column: {date_column}...")
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
        
        logger.info("Data loaded successfully.")
        return df, df.columns.tolist()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, []

# Infer dataset granularity
def infer_granularity(df):
    logger.info("Inferring dataset granularity...")
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
        logger.warning(f"Could not infer granularity: {e}. Defaulting to Weekly.")
        return ['W']

# Aggregate data
def aggregate_data(df, target_column, frequency='W', group_columns=None):
    logger.info(f"Aggregating data with target_column: {target_column}, frequency: {frequency}, group_columns: {group_columns}...")
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
            
            # Ensure continuous dates for each group
            all_dates = pd.date_range(start=df_copy['ds'].min(), end=df_copy['ds'].max(), freq=frequency)
            all_groups = df_copy[group_columns].drop_duplicates()
            
            # Create a MultiIndex with all combinations of dates and groups
            multi_index = pd.MultiIndex.from_product(
                [all_dates] + [all_groups[col] for col in group_columns],
                names=['ds'] + group_columns
            )
            full_df = pd.DataFrame(index=multi_index).reset_index()
            
            # Merge with aggregated data and fill missing values
            agg_df = full_df.merge(agg_df, on=['ds'] + group_columns, how='left')
            agg_df[target_column] = agg_df[target_column].fillna(0)  # Fill missing sales with 0
        else:
            agg_df = df_copy.groupby('ds', as_index=False)[target_column].sum()
            
            # Ensure continuous dates
            all_dates = pd.date_range(start=df_copy['ds'].min(), end=df_copy['ds'].max(), freq=frequency)
            full_df = pd.DataFrame({'ds': all_dates})
            agg_df = full_df.merge(agg_df, on='ds', how='left')
            agg_df[target_column] = agg_df[target_column].fillna(0)
        
        logger.info("Data aggregated successfully.")
        return agg_df
    except Exception as e:
        st.error(f"Error aggregating data: {e}")
        logger.error(f"Error aggregating data: {str(e)}")
        logger.error(traceback.format_exc())
        return df

# Feature engineering
def engineer_features(df, target_column, background, columns):
    logger.info(f"Running feature engineering for target_column: {target_column}...")
    try:
        if target_column in df.columns:
            st.info(f"Target column '{target_column}' already exists.")
            logger.info(f"Target column '{target_column}' already exists. Skipping feature engineering.")
            return df
        
        logger.info("Invoking feature_chain...")
        feature_code = feature_chain.invoke({
            "background": background,
            "columns": ", ".join(columns),
            "target_column": target_column
        }).content
        logger.info(f"Feature code generated: {feature_code}")
        local_vars = {'df': df.copy()}
        exec(feature_code, globals(), local_vars)
        df = local_vars['df']
        st.success(f"Created '{target_column}'.")
        logger.info(f"Created target column '{target_column}'.")
        return df
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        logger.error(f"Error in feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        return df

# Forecast for a single group
def run_forecast(df, target_column, periods, frequency, data_color, forecast_color):
    logger.info(f"Running forecast for target_column: {target_column}, periods: {periods}, frequency: {frequency}...")
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found.")
        logger.error(f"Target column '{target_column}' not found.")
        return None, None, None, None, None
    
    df_prophet = df[['ds', target_column]].rename(columns={target_column: 'y'})
    if len(df_prophet.dropna()) < 2:
        st.warning(f"Not enough data for forecasting (less than 2 non-NaN rows).")
        logger.warning("Not enough data for forecasting (less than 2 non-NaN rows).")
        return None, None, None, None, None
    
    try:
        logger.info("Fitting Prophet model...")
        model = Prophet()
        model.fit(df_prophet)
        logger.info("Prophet model fitted successfully.")
        future = model.make_future_dataframe(periods=periods, freq=frequency, include_history=True)
        logger.info("Making forecast...")
        forecast = model.predict(future)
        logger.info("Forecast generated successfully.")
        
        # Calculate appropriate figure width based on number of data points
        num_data_points = len(df_prophet) + periods
        base_width = 10
        width_factor = min(max(1, num_data_points / 100), 3)  # Limit to 3x base width
        fig_width = base_width * width_factor
        
        # Create matplotlib figure for standard display
        logger.info("Creating matplotlib figure...")
        fig1, ax1 = plt.subplots(figsize=(fig_width, 6), dpi=300)
        last_historical_date = df_prophet['ds'].max()
        historical_data = df_prophet[df_prophet['ds'] <= last_historical_date]
        forecast_data = forecast[forecast['ds'] > last_historical_date]
        
        ax1.plot(historical_data['ds'], historical_data['y'], '-', color=data_color, label='Historical Data')
        ax1.plot(forecast_data['ds'], forecast_data['yhat'], '-', color=forecast_color, label='Forecast')
        ax1.fill_between(forecast_data['ds'], forecast_data['yhat_lower'], forecast_data['yhat_upper'], color=forecast_color, alpha=0.2)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax1.set_title(f'{target_column} Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(target_column)
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        logger.info("Matplotlib figure created successfully.")
        
        # Create Plotly figure for interactive zooming
        logger.info("Creating Plotly figure...")
        plotly_fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Add historical data trace
        plotly_fig.add_trace(
            go.Scatter(
                x=historical_data['ds'],
                y=historical_data['y'],
                mode='lines',
                name='Historical Data',
                line=dict(color=data_color)
            )
        )
        
        # Add forecast trace
        plotly_fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color=forecast_color)
            )
        )
        
        # Add confidence interval
        plotly_fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
                y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba({int(forecast_color[1:3], 16)},{int(forecast_color[3:5], 16)},{int(forecast_color[5:7], 16)},0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            )
        )
        
        # Update layout
        plotly_fig.update_layout(
            title=f'{target_column} Forecast',
            xaxis_title='Date',
            yaxis_title=target_column,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            width=fig_width*100,
            height=600
        )
        
        plotly_fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        logger.info("Plotly figure created successfully.")
        
        logger.info("Generating forecast components plot...")
        fig2 = model.plot_components(forecast, figsize=(10, 8), dpi=300)
        logger.info("Forecast components plot generated.")
        
        return model, forecast, fig1, fig2, plotly_fig
    except Exception as e:
        st.error(f"Error in Prophet model: {e}")
        logger.error(f"Error in Prophet model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None

# Generate insights
def get_insights(forecast, target_column, context):
    logger.info(f"Generating insights for target_column: {target_column}...")
    try:
        # Extract key forecast data for better insights
        recent_forecast = forecast[['ds', 'yhat']].tail(10)
        forecast_trend = "increasing" if recent_forecast['yhat'].iloc[-1] > recent_forecast['yhat'].iloc[0] else "decreasing"
        percent_change = ((recent_forecast['yhat'].iloc[-1] - recent_forecast['yhat'].iloc[0]) / recent_forecast['yhat'].iloc[0] * 100) if recent_forecast['yhat'].iloc[0] != 0 else 0
        
        # Generate insights using LLM
        logger.info("Invoking forecast_chain for insights...")
        insights = forecast_chain.invoke({
            "task": "Provide detailed business insights",
            "data": f"forecast for {target_column}: {recent_forecast.to_string()}, with a {forecast_trend} trend of {percent_change:.2f}% over the forecast period",
            "context": context
        }).content
        logger.info("Insights generated successfully.")
        
        # If insights generation fails or returns empty, provide a fallback
        if not insights or len(insights.strip()) < 10:
            logger.warning("Insights generation failed or returned empty. Using fallback.")
            return f"Based on the forecast, {target_column} shows a {forecast_trend} trend with approximately {abs(percent_change):.2f}% change over the forecast period. This suggests that business planning should account for this {forecast_trend} pattern in the coming periods."
        
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        logger.error(f"Error generating insights: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Unable to generate detailed insights due to an error. However, the forecast data suggests monitoring {target_column} closely for upcoming periods as trends may impact business operations."

# Multi-group forecast with descriptive headers and filtering
def run_multi_group_forecast(df, group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=10, filter_group=None, selected_group=None):
    logger.info(f"Running multi-group forecast for group_columns: {group_columns}, target_column: {target_column}...")
    group_title = filter_group if filter_group else " & ".join(group_columns)
    
    # Generate a descriptive header based on group and context
    if filter_group:
        description = f"Analyzing {target_column} trends for top {top_n} {filter_group}s based on {context.lower()}"
    else:
        description = f"Forecasting {target_column} across top {top_n} combinations of {' and '.join(group_columns)} from {context.lower()}"
    st.subheader(description)
    
    logger.info("Aggregating data for multi-group forecast...")
    agg_df = aggregate_data(df, target_column, frequency, group_columns if not filter_group else [filter_group])
    logger.info("Data aggregated successfully.")
    
    # Calculate appropriate figure width based on data points
    max_data_points = len(agg_df) + periods
    base_width = 10
    width_factor = min(max(1, max_data_points / 100), 3)  # Limit to 3x base width
    fig_width = base_width * width_factor
    
    logger.info("Creating comparison figure...")
    fig_compare, ax_compare = plt.subplots(figsize=(fig_width, 6), dpi=300)
    forecasts_dict = {}
    agg_df_dict = {}  # Store aggregated data per group
    
    if len(group_columns if not filter_group else [filter_group]) == 1:
        group_values = agg_df[group_columns[0] if not filter_group else filter_group].value_counts().nlargest(top_n).index.tolist()
        combined_groups = [(val,) for val in group_values]
    else:
        group_sums = agg_df.groupby(group_columns)[target_column].sum().nlargest(top_n)
        combined_groups = list(group_sums.index)
    
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
        
        logger.info(f"Processing group: {group_label}...")
        if not group_data.empty and len(group_data.dropna()) >= 2:
            try:
                prophet_data = group_data[['ds', target_column]].rename(columns={target_column: 'y'})
                logger.info(f"Fitting Prophet model for group: {group_label}...")
                model = Prophet()
                model.fit(prophet_data)
                logger.info(f"Prophet model fitted for group: {group_label}.")
                future = model.make_future_dataframe(periods=periods, freq=frequency, include_history=True)  # Include history
                logger.info(f"Generating forecast for group: {group_label}...")
                forecast = model.predict(future)
                logger.info(f"Forecast generated for group: {group_label}.")
                
                last_date = prophet_data['ds'].max()
                historical_data = prophet_data[prophet_data['ds'] <= last_date]
                forecast_full = forecast[forecast['ds'] >= last_date]  # Include the last historical date
                
                # Combine historical and forecast data for a continuous line
                combined_data = pd.concat([
                    historical_data.rename(columns={'y': 'value'}),
                    forecast_full[['ds', 'yhat']].rename(columns={'yhat': 'value'})
                ]).drop_duplicates(subset='ds', keep='first')
                
                # Plot only if selected_group matches or no filter is applied
                if selected_group is None or group_label == selected_group:
                    # Plot historical part with lower alpha
                    ax_compare.plot(
                        historical_data['ds'], 
                        historical_data['y'], 
                        '-', 
                        color=group_color, 
                        alpha=0.5, 
                        label=f"{group_label} (Historical)"
                    )
                    # Plot the combined line (historical + forecast) to ensure continuity
                    ax_compare.plot(
                        combined_data['ds'], 
                        combined_data['value'], 
                        '-', 
                        color=group_color, 
                        label=f"{group_label} (Forecast)"
                    )
                
                forecasts_dict[group_label] = forecast
                agg_df_dict[group_label] = group_data  # Store for heatmap and detailed view
                logger.info(f"Group {group_label} processed successfully.")
            except Exception as e:
                st.warning(f"Could not forecast for {group_label}: {e}")
                logger.warning(f"Could not forecast for {group_label}: {str(e)}")
                logger.warning(traceback.format_exc())
        else:
            st.warning(f"Skipping {group_label}: Not enough data (less than 2 non-NaN rows).")
            logger.warning(f"Skipping {group_label}: Not enough data (less than 2 non-NaN rows).")
    
    ax_compare.set_title(f"{target_column} Forecast Comparison")
    ax_compare.set_xlabel("Date")
    ax_compare.set_ylabel(target_column)
    ax_compare.grid(True, linestyle='--', alpha=0.7)
    ax_compare.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.gcf().autofmt_xdate()  # Better date formatting on x-axis
    plt.tight_layout()
    logger.info("Comparison figure created successfully.")
    
    if forecasts_dict:
        group_labels = list(forecasts_dict.keys())
        st.session_state[f"heatmap_data_{group_title}"] = (forecasts_dict, agg_df_dict, group_labels, target_column, group_title)
        logger.info(f"Stored heatmap data for {group_title}.")
    else:
        st.error("No valid forecasts generated for any group.")
        logger.error("No valid forecasts generated for any group.")
    
    return fig_compare, forecasts_dict, agg_df

# Dynamic Heatmap with Filtering
def create_forecast_heatmap(forecasts_dict=None, group_labels=None, target_column=None, group_title=None, agg_df_dict=None):
    logger.info(f"Creating forecast heatmap for group_title: {group_title}...")
    if not all([forecasts_dict, group_labels, target_column, group_title]):
        st.warning("Missing data for heatmap generation.")
        logger.warning("Missing data for heatmap generation.")
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
        logger.warning("Please select at least one group to display the heatmap.")
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
                logger.warning(f"No future data for group {group_label} after {last_historical_date}")
    
    if all_forecasts.empty:
        st.warning("No future forecast data available for the selected groups.")
        logger.warning("No future forecast data available for the selected groups.")
        return None
    
    pivot_df = all_forecasts.pivot(index='group', columns='ds', values='yhat')
    pivot_df.columns = pivot_df.columns.strftime('%Y-%m-%d')
    
    logger.info("Creating heatmap figure...")
    fig, ax = plt.subplots(figsize=(14, len(selected_groups) * 0.5 + 2), dpi=300)
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5, ax=ax)
    ax.set_title(f"{target_column} Forecast Heatmap by {group_title}")
    ax.set_ylabel("Group")
    ax.set_xlabel("Date")
    plt.tight_layout()
    logger.info("Heatmap figure created successfully.")
    
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
    logger.info("Converting DataFrame to CSV...")
    return df.to_csv(index=False)