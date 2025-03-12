import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from prophet import Prophet
import numpy as np

# Load environment variables
load_dotenv()

def load_data(file_path, date_column='Date'):
    """
    Load the dataset for testing
    """
    try:
        df = pd.read_csv(file_path)
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        
        df = df.rename(columns={date_column: 'ds'})
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        if df['ds'].isna().any():
            raise ValueError("Some dates could not be parsed.")
        
        print(f"Loaded DataFrame columns: {df.columns.tolist()}")
        return df, df.columns.tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, []

def aggregate_data(df, target_column, frequency='W', group_columns=None):
    """
    Aggregate data based on frequency and grouping
    """
    try:
        df_copy = df.copy()
        if frequency == 'D':
            df_copy['ds'] = df_copy['ds'].dt.floor('D')
        elif frequency == 'W':
            df_copy['ds'] = df_copy['ds'].dt.to_period('W').dt.to_timestamp()
        elif frequency == 'ME':
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
        print(f"Error aggregating data: {e}")
        return df

def run_forecast(df, target_column, periods, frequency='W', data_color='#1E88E5', forecast_color='#FFC107'):
    """
    Run a Prophet forecast on the provided DataFrame
    """
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        return None, None, None, None
    
    df_prophet = df[['ds', target_column]].rename(columns={target_column: 'y'})
    if len(df_prophet.dropna()) < 2:
        print(f"Not enough data for forecasting (less than 2 non-NaN rows).")
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
        print(f"Error in Prophet model: {e}")
        return None, None, None, None

def test_forecast_results(file_path):
    """
    Test the forecasting functionality with a sample dataset.
    
    Args:
        file_path: Path to the CSV file to test
    """
    # Load the dataset
    df, columns = load_data(file_path, date_column='Date')
    if df is None:
        print("Failed to load data. Please check the file path and format.")
        return
    
    print("Dataset Sample:")
    print(df.head())
    
    # Define test parameters
    target_column = 'Weekly_Sales'
    periods = 12  # 12 weeks forecast
    frequency = 'W'
    data_color = '#1E88E5'
    forecast_color = '#FFC107'

    # Test 1: All Data Combined
    print("\nTesting All Data Combined:")
    all_agg_df = aggregate_data(df, target_column, frequency)
    model, forecast, fig1, fig2 = run_forecast(all_agg_df, target_column, periods, frequency, data_color, forecast_color)
    if forecast is not None:
        print("\nAll Data Forecast (Last 5 rows):")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        # Validation: Check for negative sales
        if (forecast['yhat'] < 0).any():
            print("Warning: Negative sales detected in forecast!")
        else:
            print("Validation: No negative sales detected.")
        
        # Save the plots
        os.makedirs('models', exist_ok=True)
        plt.figure(fig1.number)
        plt.savefig(os.path.join('models', 'all_data_forecast.png'))
        plt.close(fig1)
        if fig2:
            plt.figure(fig2.number)
            plt.savefig(os.path.join('models', 'all_data_components.png'))
            plt.close(fig2)
    
    # Test 2: Grouped by Store=1
    print("\nTesting Store=1:")
    store_1_df = aggregate_data(df[df['Store'] == 1], target_column, frequency)
    model, forecast, fig1, fig2 = run_forecast(store_1_df, target_column, periods, frequency, data_color, forecast_color)
    if forecast is not None:
        print("\nStore 1 Forecast (Last 5 rows):")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        # Validation: Compare with historical max
        historical_max = store_1_df[target_column].max()
        forecast_max = forecast['yhat'].max()
        print(f"Historical Max Sales: {historical_max:.2f}, Forecast Max: {forecast_max:.2f}")
        if forecast_max < 0:
            print("Warning: Negative sales detected in Store 1 forecast!")
        
        plt.figure(fig1.number)
        plt.savefig(os.path.join('models', 'store_1_forecast.png'))
        plt.close(fig1)
        if fig2:
            plt.figure(fig2.number)
            plt.savefig(os.path.join('models', 'store_1_components.png'))
            plt.close(fig2)
    
    # Test 3: Grouped by Dept=1
    print("\nTesting Dept=1:")
    dept_1_df = aggregate_data(df[df['Dept'] == 1], target_column, frequency)
    model, forecast, fig1, fig2 = run_forecast(dept_1_df, target_column, periods, frequency, data_color, forecast_color)
    if forecast is not None:
        print("\nDept 1 Forecast (Last 5 rows):")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        # Validation: Check trend direction
        historical_trend = dept_1_df[target_column].iloc[-1] - dept_1_df[target_column].iloc[0]
        forecast_trend = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]
        print(f"Historical Trend: {historical_trend:.2f}, Forecast Trend: {forecast_trend:.2f}")
        if (forecast['yhat'] < 0).any():
            print("Warning: Negative sales detected in Dept 1 forecast!")
        
        plt.figure(fig1.number)
        plt.savefig(os.path.join('models', 'dept_1_forecast.png'))
        plt.close(fig1)
        if fig2:
            plt.figure(fig2.number)
            plt.savefig(os.path.join('models', 'dept_1_components.png'))
            plt.close(fig2)

    print("\nReview the outputs and plots to confirm correctness!")
    print(f"Plots have been saved to the 'models' directory.")

if __name__ == "__main__":
    # Use a relative path to the dataset
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'WalmartStoreSalesForecasting')
    file_path = os.path.join(data_dir, 'train.csv')
    
    if os.path.exists(file_path):
        test_forecast_results(file_path)
    else:
        print(f"Error: File not found at {file_path}")
        print("Please make sure the dataset exists in the data/WalmartStoreSalesForecasting directory.")
