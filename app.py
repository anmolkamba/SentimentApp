import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys 

from data_fetcher import DataFetcher
from technical_analysis import TechnicalAnalyzer
from ml_predictor import MLPredictor
from sentiment_analyzer import SentimentAnalyzer


def plot_price_data(df, title="Price Data"):
    """Create an interactive price chart"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    
    return fig

def plot_technical_indicators(df):
    """Create technical indicators chart"""
    fig = go.Figure()
    
    # Add RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ))
    
    # Add MACD
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ))
    
    # fig.add_trace(go.Scatter(
    #     x=df.index,
    #     y=df['MACD_Signal'],
    #     name='MACD Signal',
    #     line=dict(color='orange')
    # ))
    
    fig.update_layout(
        title="Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )
    
    return fig

def plot_predictions(historical_data, predictions, title="Price Predictions"):
    """Create prediction chart"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add predictions
    future_dates = pd.date_range(
        start=historical_data.index[-1] + timedelta(days=1),
        periods=len(predictions['random_forest']),
        freq='D'
    )
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions['random_forest'],
        name='Random Forest',
        line=dict(color='green', dash='dash')
    ))
    
    if predictions['lstm'] is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions['lstm'],
            name='LSTM',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    
    return fig

def main():
    st.set_page_config(page_title="ETF Price Predictor", layout="wide")
    st.title("ETF Price Predictor and Sentiment Analyzer")
    
    # Sidebar
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("ETF Symbol", "QQQ")
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Fetch data
    with st.spinner("Fetching data..."):
        df = data_fetcher.get_stock_data(symbol, period)
        if df is None:
            st.error("Failed to fetch data. Please check the symbol and try again.")
            return
        
        # Calculate technical indicators
        df = data_fetcher.get_technical_indicators(df)
        
        # Fetch social media data
        social_data = data_fetcher.get_social_media_data(symbol)
    
    # print(df.columns)
    # sys.exit("File Error") #exit with a string message
    # Display price chart
    st.plotly_chart(plot_price_data(df, f"{symbol} Price Data"), use_container_width=True)
    
    # Technical Analysis
    st.header("Technical Analysis")
    st.plotly_chart(plot_technical_indicators(df), use_container_width=True)
    
    # Machine Learning Predictions
    st.header("Price Predictions")
    
    # Initialize ML predictor
    ml_predictor = MLPredictor(df)
    
    # Train models
    with st.spinner("Training models..."):
        rf_metrics = ml_predictor.train_random_forest()
        lstm_metrics = ml_predictor.train_lstm()
        
        if rf_metrics and lstm_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Random Forest R² Score", f"{rf_metrics['r2']:.3f}")
            with col2:
                st.metric("LSTM R² Score", f"{lstm_metrics['r2']:.3f}")
            
            # Generate predictions
            predictions = ml_predictor.predict_price(days=5)
            if predictions:
                st.plotly_chart(
                    plot_predictions(df, predictions),
                    use_container_width=True
                )
    
    # Sentiment Analysis
    if social_data is not None and not social_data.empty:
        st.header("Sentiment Analysis")
        
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer(social_data)
        
        # Analyze sentiment
        sentiment_df = sentiment_analyzer.analyze_sentiment()
        sentiment_summary = sentiment_analyzer.get_sentiment_summary()
        sentiment_trend = sentiment_analyzer.get_sentiment_trend()
        impact_score = sentiment_analyzer.get_impact_score()
        keywords = sentiment_analyzer.get_keywords()
        
        if sentiment_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Sentiment",
                    f"{sentiment_summary['average_polarity']:.2f}"
                )
            with col2:
                st.metric(
                    "Sentiment Impact Score",
                    f"{impact_score:.2f}"
                )
            with col3:
                st.metric(
                    "Subjectivity",
                    f"{sentiment_summary['average_subjectivity']:.2f}"
                )
            
            # Display sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_dist = sentiment_summary['sentiment_distribution']
            st.bar_chart(sentiment_dist)
            
            # Display keywords
            if keywords:
                st.subheader("Top Keywords")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Top Phrases:")
                    for phrase, count in keywords['top_phrases']:
                        st.write(f"- {phrase}: {count}")
                with col2:
                    st.write("Top Words:")
                    for word, count in keywords['top_words']:
                        st.write(f"- {word}: {count}")

if __name__ == "__main__":
    main() 