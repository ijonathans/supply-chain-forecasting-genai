import streamlit as st
import os
import sys
import traceback
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a title to the app
st.title("Supply Chain Forecasting with GenAI")

# Add app description and author information
st.markdown("""
<div style="text-align: left; color: #4d4d4d; padding: 10px; margin-bottom: 20px;">
    <p>Created by Ignatius Jonathan Sugijono</p>
    <p>Version 1.1</p>
    <p>Built using Streamlit, Pandas, and Plotly for interactive visualizations.</p>
    <p>For feedback, contact: ignatiusjonathan07@gmail.com</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI API key is required to use this application.")
    st.info("Please provide your OpenAI API key below. It will only be stored for this session.")
    st.code("""
1. Get your API key from https://platform.openai.com/account/api-keys
2. Enter it below
3. Your key will only be stored for this session
4. For production, set OPENAI_API_KEY in your environment variables
    """)
    api_key = st.text_input("Enter your OpenAI API key", type="password", key="api_key_input")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set successfully! The app will now load.")
    else:
        st.stop()

# Add API key input in the sidebar
api_key = st.sidebar.text_input("OpenAI API Key (required for insights)", type="password", help="Enter your OpenAI API key to enable business insights")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API key set for this session!")

try:
    # Import app.py module and all its functions
    import app
    from app import (
        load_data, 
        engineer_features, 
        aggregate_data, 
        run_forecast, 
        run_multi_group_forecast, 
        get_insights, 
        create_forecast_heatmap, 
        convert_df_to_csv
    )
    
    # Initialize session state
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None

    if 'date_column' not in st.session_state:
        st.session_state.date_column = ""

    # Main content area
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = {}
    
    # Sidebar configuration
    st.sidebar.title("Supply Chain Forecasting")
    st.sidebar.markdown("Configure your forecast parameters below:")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your time series data (CSV)", type=["csv"], key="csv_uploader")
    
    # Date column selection
    date_column = st.sidebar.text_input(
        "Date column name", 
        value=st.session_state.date_column if st.session_state.date_column else "Date", 
        key="date_column_input"
    )
    # Update session state with the current value
    st.session_state.date_column = date_column
    
    # Context input
    context = st.sidebar.text_area("Dataset context (for better feature engineering)", 
                                  "This is a supply chain dataset with sales data across different stores and departments.",
                                  key="context_input")
    
    # Target column
    target_column = st.sidebar.text_input("Target column to forecast", "Weekly_Sales", key="target_column_input")
    
    # Group columns
    group_column_input = st.sidebar.text_input("Group columns (comma-separated)", "Store,Dept", key="group_columns_input")
    selected_group_columns = [col.strip() for col in group_column_input.split(",")] if group_column_input else []
    
    # Forecast parameters
    periods = st.sidebar.slider("Forecast periods", 1, 52, 12, key="periods_slider")
    frequency_options = ["D", "W", "M"]  # Changed 'ME' to 'M' for Prophet compatibility
    frequency = st.sidebar.selectbox("Frequency", frequency_options, index=1, key="frequency_select")
    
    # Colors
    data_color = st.sidebar.color_picker("Historical data color", "#1f77b4", key="data_color_picker")
    forecast_color = st.sidebar.color_picker("Forecast color", "#ff7f0e", key="forecast_color_picker")
    
    # Run button
    run_button = st.sidebar.button("Generate Forecast", key="generate_forecast_button")
    
    # Load data
    df = None
    columns = []
    if uploaded_file:
        # Always read and display the original columns first
        original_df = pd.read_csv(uploaded_file)
        original_columns = original_df.columns.tolist()
        
        # Display column names in the main area (using original column names)
        st.subheader("Dataset Columns")
        col_display = st.expander("Click to view all columns in the dataset", expanded=True)
        with col_display:
            # Create a more visually appealing display of columns with original names
            col_data = {"Column Name": original_columns, "Sample Values": [str(original_df[col].iloc[0]) for col in original_columns]}
            st.dataframe(pd.DataFrame(col_data), use_container_width=True)
            
            # Add a copy button for convenience
            st.code(", ".join(original_columns), language="text")
            st.caption("Use the column names above for configuring your forecast parameters")
        
        # Now try to process the data with the selected date column
        try:
            # Reset the file pointer to the beginning for the actual processing
            uploaded_file.seek(0)
            
            # Process the data for forecasting
            df, columns = load_data(uploaded_file, date_column)
            st.sidebar.success(f"Data loaded successfully: {len(df)} rows, {len(columns)} columns")
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg and date_column:
                st.error(f"Error: Date column '{date_column}' not found. Please select a valid date column from the list above and update the 'Date Column' field in the sidebar.")
                
                # Add a more prominent date column selector
                st.warning("Please select the correct date column from the dropdown below:")
                new_date_column = st.selectbox("Select Date Column", options=original_columns, key="date_column_correction")
                if st.button("Update Date Column"):
                    st.session_state.date_column = new_date_column
                    st.experimental_rerun()
            else:
                st.error(f"Error loading data: {error_msg}")
                st.info("Please check your data format and try again. Make sure to select the correct date column from the list above.")
            
            # Set df to None to prevent forecast generation
            df = None
    
    # Main content area
    if df is None and not st.session_state.forecast_results:
        st.info("Please upload a CSV file with time series data to begin.")
        
        # Sample data description
        st.markdown("### Sample Data Format")
        st.markdown("""
        Your CSV file should contain at least:
        - A date column (can be renamed using the 'Date column name' field)
        - A target column to forecast (e.g., sales, demand, inventory)
        - Optional: Group columns for segmented forecasting (e.g., store, department, product)
        """)
        
        # Example usage
        st.markdown("### Example Usage")
        st.markdown("""
        1. Upload your CSV file
        2. Specify the date column name
        3. Provide context about your dataset
        4. Enter the target column to forecast
        5. Optionally add group columns for segmented analysis
        6. Set forecast periods and frequency
        7. Click 'Generate Forecast'
        """)
    
    # Run forecasting when button is clicked
    if run_button and df is not None:
        with st.spinner("Generating forecasts..."):
            try:
                df = engineer_features(df, target_column, context, columns)
                
                # Create a placeholder for results
                results_dict = {
                    'df': df,
                    'target_column': target_column,
                    'periods': periods,
                    'frequency': frequency,
                    'context': context
                }
                
                # Single forecast (no grouping)
                if not selected_group_columns or not selected_group_columns[0]:
                    agg_df = aggregate_data(df, target_column, frequency)
                    model, forecast, fig1, fig2 = run_forecast(agg_df, target_column, periods, frequency, data_color, forecast_color)
                    if model and forecast is not None:
                        results_dict['single'] = (fig1, forecast)
                        st.session_state.forecast_results = results_dict
                        st.experimental_rerun()
                
                # Multi-group forecasting
                else:
                    # Primary group forecasting
                    primary_fig, primary_forecasts, primary_agg_df = run_multi_group_forecast(
                        df, [selected_group_columns[0]], target_column, periods, frequency, 
                        context, data_color, forecast_color, top_n=10
                    )
                    results_dict['primary'] = (primary_fig, primary_forecasts, primary_agg_df)
                    
                    # Secondary group forecasting (if available)
                    if len(selected_group_columns) >= 2:
                        secondary_fig, secondary_forecasts, secondary_agg_df = run_multi_group_forecast(
                            df, [selected_group_columns[1]], target_column, periods, frequency, 
                            context, data_color, forecast_color, top_n=10
                        )
                        results_dict['secondary'] = (secondary_fig, secondary_forecasts, secondary_agg_df)
                    
                    # Combined group forecasting
                    combined_fig, combined_forecasts, combined_agg_df = run_multi_group_forecast(
                        df, selected_group_columns, target_column, periods, frequency, 
                        context, data_color, forecast_color, top_n=10
                    )
                    results_dict['combined'] = (combined_fig, combined_forecasts, combined_agg_df)
                    
                    st.session_state.forecast_results = results_dict
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error in forecast generation: {str(e)}")
                st.code(traceback.format_exc())
    
    # Display results if available
    if 'forecast_results' in st.session_state and st.session_state.forecast_results:
        results = st.session_state.forecast_results
        df = results['df']
        target_column = results['target_column']
        periods = results['periods']
        frequency = results.get('frequency', 'W')
        context = results.get('context', '')
        
        # Filter options
        filter_options = []
        if 'single' in results:
            filter_options.append("Overall (No Grouping)")
        
        if selected_group_columns:
            if 'combined' in results:
                filter_options.append(f"Combined ({' & '.join(selected_group_columns)})")
            if 'primary' in results:
                filter_options.append(selected_group_columns[0])
            if 'secondary' in results and len(selected_group_columns) >= 2:
                filter_options.append(selected_group_columns[1])
        
        selected_filter = st.selectbox("View forecast by", filter_options, key="view_forecast_select")
        
        # Display based on filter selection
        if selected_filter == "Overall (No Grouping)" and 'single' in results:
            fig1, forecast = results['single']
            st.subheader("Overall Forecast Results")
            if fig1:
                st.pyplot(fig1)
            
            # Add business insights section
            st.subheader("Business Insights")
            
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable AI-powered business insights")
            else:
                with st.spinner("Generating business insights..."):
                    insights = get_insights(forecast, target_column, context)
                    st.markdown("""
                    <div style="background-color:#f0f7fb; padding:15px; border-radius:10px; border-left:5px solid #2196F3;">
                        <h4 style="color:#1976D2;">AI-Generated Business Recommendations</h4>
                        <p>{}</p>
                    </div>
                    """.format(insights), unsafe_allow_html=True)
            
            if forecast is not None:
                st.subheader("Forecast Details")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                
                # Download button
                st.download_button(
                    label="Download Forecast CSV",
                    data=convert_df_to_csv(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]),
                    file_name=f"forecast_{target_column}.csv",
                    mime="text/csv",
                    key="download_overall_forecast"
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
                    st.warning("No comparison plot generated for Combined view.")
                
                if combined_forecasts:
                    group_title = " & ".join(selected_group_columns)
                    group_labels = list(combined_forecasts.keys())
                    
                    # Display heatmap first
                    if f"heatmap_data_{group_title}" in st.session_state:
                        forecasts_dict, agg_df_dict, all_labels, target_col, title = st.session_state[f"heatmap_data_{group_title}"]
                        heatmap_fig = create_forecast_heatmap(forecasts_dict, all_labels, target_col, title, agg_df_dict)
                        if not heatmap_fig:
                            st.warning("Heatmap generation failed.")
                    
                    # Move the selection box below the heatmap
                    selected_group = st.selectbox(
                        f"Select {group_title} to view the insights",
                        options=["All"] + group_labels,
                        key="combined_group_select"
                    )
                    
                    all_forecasts = pd.concat([forecast[['ds', 'yhat']].assign(group=group) for group, forecast in combined_forecasts.items()])
                    st.download_button(
                        label=f"Download {group_title} Forecasts",
                        data=convert_df_to_csv(all_forecasts),
                        file_name=f"{group_title}_forecasts_{target_column}.csv",
                        mime="text/csv",
                        key="download_combined_forecasts"
                    )
                    
                    if selected_group != "All":
                        st.subheader(f"Detailed Forecast for {selected_group}")
                        forecast = combined_forecasts[selected_group]
                        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                        
                        # Add business insights section
                        st.subheader(f"Business Insights for {selected_group}")
                        
                        if not os.getenv("OPENAI_API_KEY"):
                            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable AI-powered business insights")
                        else:
                            with st.spinner("Generating business insights..."):
                                group_insights = get_insights(forecast, target_column, f"{context} for {selected_group}")
                                st.markdown("""
                                <div style="background-color:#f0f7fb; padding:15px; border-radius:10px; border-left:5px solid #2196F3;">
                                    <h4 style="color:#1976D2;">AI-Generated Business Recommendations</h4>
                                    <p>{}</p>
                                </div>
                                """.format(group_insights), unsafe_allow_html=True)
                        
                        group_data = combined_agg_df
                        parts = selected_group.split(' & ')
                        for i, part in enumerate(parts):
                            col_val = part.split('=')
                            if len(col_val) == 2:
                                group_data = group_data[group_data[selected_group_columns[i]] == col_val[1]]
                        
                        st.write(f"Detailed view for {selected_group} has {len(group_data)} rows")
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
                    
                    # Display heatmap first
                    if f"heatmap_data_{group_title}" in st.session_state:
                        forecasts_dict, agg_df_dict, all_labels, target_col, title = st.session_state[f"heatmap_data_{group_title}"]
                        heatmap_fig = create_forecast_heatmap(forecasts_dict, all_labels, target_col, title, agg_df_dict)
                        if not heatmap_fig:
                            st.warning("Heatmap generation failed.")
                    
                    # Move the selection box below the heatmap
                    selected_group = st.selectbox(
                        f"Select {selected_group_columns[0]} to view",
                        options=["All"] + group_labels,
                        key="primary_group_select"
                    )
                    
                    all_forecasts = pd.concat([forecast[['ds', 'yhat']].assign(group=group) for group, forecast in primary_forecasts.items()])
                    st.download_button(
                        label=f"Download {selected_group_columns[0]} Forecasts",
                        data=convert_df_to_csv(all_forecasts),
                        file_name=f"{selected_group_columns[0]}_forecasts_{target_column}.csv",
                        mime="text/csv",
                        key="download_primary_forecasts"
                    )
                    
                    if selected_group != "All":
                        st.subheader(f"Detailed Forecast for {selected_group}")
                        forecast = primary_forecasts[selected_group]
                        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                        
                        # Add business insights section
                        st.subheader(f"Business Insights for {selected_group}")
                        
                        if not os.getenv("OPENAI_API_KEY"):
                            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable AI-powered business insights")
                        else:
                            with st.spinner("Generating business insights..."):
                                group_insights = get_insights(forecast, target_column, f"{context} for {selected_group}")
                                st.markdown("""
                                <div style="background-color:#f0f7fb; padding:15px; border-radius:10px; border-left:5px solid #2196F3;">
                                    <h4 style="color:#1976D2;">AI-Generated Business Recommendations</h4>
                                    <p>{}</p>
                                </div>
                                """.format(group_insights), unsafe_allow_html=True)
                        
                        group_data = primary_agg_df[primary_agg_df[selected_group_columns[0]] == selected_group]
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
                    
                    # Display heatmap first
                    if f"heatmap_data_{group_title}" in st.session_state:
                        forecasts_dict, agg_df_dict, all_labels, target_col, title = st.session_state[f"heatmap_data_{group_title}"]
                        heatmap_fig = create_forecast_heatmap(forecasts_dict, all_labels, target_col, title, agg_df_dict)
                        if not heatmap_fig:
                            st.warning("Heatmap generation failed.")
                    
                    # Move the selection box below the heatmap
                    selected_group = st.selectbox(
                        f"Select {selected_group_columns[1]} to view",
                        options=["All"] + group_labels,
                        key="secondary_group_select"
                    )
                    
                    all_forecasts = pd.concat([forecast[['ds', 'yhat']].assign(group=group) for group, forecast in secondary_forecasts.items()])
                    st.download_button(
                        label=f"Download {selected_group_columns[1]} Forecasts",
                        data=convert_df_to_csv(all_forecasts),
                        file_name=f"{selected_group_columns[1]}_forecasts_{target_column}.csv",
                        mime="text/csv",
                        key="download_secondary_forecasts"
                    )
                    
                    if selected_group != "All":
                        st.subheader(f"Detailed Forecast for {selected_group}")
                        forecast = secondary_forecasts[selected_group]
                        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                        
                        # Add business insights section
                        st.subheader(f"Business Insights for {selected_group}")
                        
                        if not os.getenv("OPENAI_API_KEY"):
                            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable AI-powered business insights")
                        else:
                            with st.spinner("Generating business insights..."):
                                group_insights = get_insights(forecast, target_column, f"{context} for {selected_group}")
                                st.markdown("""
                                <div style="background-color:#f0f7fb; padding:15px; border-radius:10px; border-left:5px solid #2196F3;">
                                    <h4 style="color:#1976D2;">AI-Generated Business Recommendations</h4>
                                    <p>{}</p>
                                </div>
                                """.format(group_insights), unsafe_allow_html=True)
                        
                        group_data = secondary_agg_df[secondary_agg_df[selected_group_columns[1]] == selected_group]
                        st.write(f"Detailed view for {selected_group} has {len(group_data)} rows")
                        st.write(f"Group data sample:\n{group_data.head()}")
                        
                        model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                        if fig1:
                            st.pyplot(fig1)
                        if fig2:
                            st.pyplot(fig2)
    
    # Reset button
    if st.session_state.forecast_results:
        if st.sidebar.button("Reset Application", key="reset_app_button"):
            st.session_state.forecast_results = {}
            st.experimental_rerun()
            
except Exception as e:
    st.error(f"Error loading the application: {str(e)}")
    st.code(traceback.format_exc())
    st.info("Please check the error message above and ensure all dependencies are installed correctly.")

# This file serves as an entry point for Streamlit Cloud
# It will automatically use the app.py file for the main functionality
