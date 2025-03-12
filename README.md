# Supply Chain Time Series Forecasting

A powerful time series forecasting application for supply chain management, leveraging Prophet and Generative AI for insights.

## Features

- **Time Series Forecasting**: Utilizes Facebook Prophet for accurate time series forecasting
- **AI-Powered Insights**: Generates business insights based on forecast results using OpenAI
- **Multi-dimensional Analysis**: Analyze forecasts across different dimensions (e.g., Store, Department)
- **Dynamic Heatmap**: Visualize forecast results across different groups
- **Interactive UI**: Built with Streamlit for an intuitive user experience

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/supply-chain-forecasting.git
   cd supply-chain-forecasting
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app locally:
   ```
   streamlit run app.py
   ```

2. Upload your CSV file containing time series data
3. Select the date column and target variable to forecast
4. Configure additional parameters like forecast periods and grouping dimensions
5. Click "Generate Forecast" to see the results

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your GitHub repository
4. Add your OpenAI API key as a secret in the Streamlit Cloud dashboard

### Vercel Deployment Notes

When deploying to Vercel:
- There's a 250MB unzipped size limit for serverless functions
- Use minimal dependencies in requirements.txt
- Consider using Vercel's Edge Functions for Python code
- Add environment variables in Vercel dashboard instead of .env files

## Sample Datasets

The application works with any time series data in CSV format. Sample datasets are included in the `data` directory:
- Walmart Store Sales Forecasting
- Dunn Humby Breakfast at the Frat

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── .env                    # Environment variables (not included in repo)
├── requirements.txt        # Project dependencies
├── .streamlit/             # Streamlit configuration
│   └── config.toml         # Streamlit settings
├── data/                   # Sample datasets
│   ├── WalmartStoreSalesForecasting/
│   └── DunnHumby_BreakfastatTheFrat/
└── README.md               # Project documentation
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
