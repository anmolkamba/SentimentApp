# ETF Price Predictor and Sentiment Analyzer

This application provides comprehensive analysis for ETFs (including QQQ) using:
- Technical Analysis
- Machine Learning Predictions
- Social Media and News Sentiment Analysis

## Features
- Price prediction using technical indicators
- Machine learning-based price forecasting
- Sentiment analysis from Reddit and financial news
- Interactive visualization of predictions and analysis
- Support for multiple ETFs (QQQ, SPY, etc.)

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_app_name
   NEWS_API_KEY=your_news_api_key
   ```

## API Setup Instructions

### Reddit API
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..."
3. Select "script"
4. Fill in the required information
5. After creation, you'll get the client ID and client secret

### NewsAPI
1. Go to https://newsapi.org/
2. Sign up for a free account
3. Get your API key from the dashboard

## Usage
Run the application:
```bash
streamlit run app.py
```

## Project Structure
- `app.py`: Main application file
- `technical_analysis.py`: Technical analysis module
- `ml_predictor.py`: Machine learning prediction module
- `sentiment_analyzer.py`: Sentiment analysis module
- `data_fetcher.py`: Data fetching module 