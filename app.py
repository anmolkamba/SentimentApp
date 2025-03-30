import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys 
import yfinance as yf

from data_fetcher import DataFetcher
from technical_analysis import TechnicalAnalyzer
from ml_predictor import MLPredictor
from sentiment_analyzer import SentimentAnalyzer

# List of popular ETFs to track with their industries
POPULAR_ETFS = {
    "QQQ": "Technology",
    "SPY": "Broad Market",
    "VOO": "Broad Market",
    "VTI": "Broad Market",
    "BND": "Fixed Income",
    "VEA": "International",
    "VWO": "International",
    "AGG": "Fixed Income",
    "GLD": "Commodity",
    "TLT": "Fixed Income",
    "IWM": "Small Cap",
    "DIA": "Broad Market",
    "XLK": "Technology",
    "XLF": "Financial",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrial",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials"
}

def calculate_etf_performance(etf_list, period='1y'):
    """Calculate performance metrics for a list of ETFs"""
    data_fetcher = DataFetcher()
    performance_data = []
    
    for etf in etf_list:
        try:
            df = data_fetcher.get_stock_data(etf, period)
            if df is not None and not df.empty:
                # Calculate performance metrics
                current_price = df['Close'].iloc[-1]
                start_price = df['Close'].iloc[0]
                performance = ((current_price - start_price) / start_price) * 100
                volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
                
                # Get ETF info
                etf_info = yf.Ticker(etf).info
                description = etf_info.get('longBusinessSummary', 'No description available')
                
                performance_data.append({
                    'ETF': etf,
                    'Industry': POPULAR_ETFS.get(etf, 'Unknown'),
                    'Description': description,
                    'Performance (%)': round(performance, 2),
                    'Volatility (%)': round(volatility, 2),
                    'Current Price': round(current_price, 2),
                    'Start Price': round(start_price, 2)
                })
        except Exception as e:
            st.warning(f"Could not fetch data for {etf}: {str(e)}")
    
    return pd.DataFrame(performance_data)

def display_etf_performance():
    """Display top and bottom performing ETFs"""
    st.header("ETF Performance Analysis")
    
    # Add period selector
    period = st.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,
        key="performance_period"
    )
    
    with st.spinner("Calculating ETF performance..."):
        performance_df = calculate_etf_performance(POPULAR_ETFS.keys(), period)
        
        if not performance_df.empty:
            # Sort by performance
            performance_df = performance_df.sort_values('Performance (%)', ascending=False)
            
            # Display top 5 performers
            st.subheader("Top 5 Performing ETFs")
            top_5 = performance_df.head()
            top_5_display = top_5[[
                'ETF', 'Industry', 'Performance (%)', 
                'Volatility (%)', 'Current Price', 'Start Price'
            ]]
            st.dataframe(
                top_5_display.style.format({
                    'Performance (%)': '{:.2f}%',
                    'Volatility (%)': '{:.2f}%',
                    'Current Price': '${:.2f}',
                    'Start Price': '${:.2f}'
                }),
                use_container_width=True
            )
            
            # Display bottom 5 performers
            st.subheader("Bottom 5 Performing ETFs")
            bottom_5 = performance_df.tail()
            bottom_5_display = bottom_5[[
                'ETF', 'Industry', 'Performance (%)', 
                'Volatility (%)', 'Current Price', 'Start Price'
            ]]
            st.dataframe(
                bottom_5_display.style.format({
                    'Performance (%)': '{:.2f}%',
                    'Volatility (%)': '{:.2f}%',
                    'Current Price': '${:.2f}',
                    'Start Price': '${:.2f}'
                }),
                use_container_width=True
            )
            
            # Create performance comparison chart
            fig = go.Figure()
            
            # Add top performers
            fig.add_trace(go.Bar(
                x=top_5['ETF'],
                y=top_5['Performance (%)'],
                name='Top Performers',
                marker_color='green'
            ))
            
            # Add bottom performers
            fig.add_trace(go.Bar(
                x=bottom_5['ETF'],
                y=bottom_5['Performance (%)'],
                name='Bottom Performers',
                marker_color='red'
            ))
            
            fig.update_layout(
                title=f"ETF Performance Comparison ({period})",
                xaxis_title="ETF",
                yaxis_title="Performance (%)",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_etf_comparison():
    """Display comprehensive ETF comparison"""
    st.header("ETF Performance Comparison")
    
    # Add period selector
    period = st.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,
        key="comparison_period"
    )
    
    with st.spinner("Calculating ETF performance..."):
        performance_df = calculate_etf_performance(POPULAR_ETFS.keys(), period)
        
        if not performance_df.empty:
            # Sort by performance
            performance_df = performance_df.sort_values('Performance (%)', ascending=False)
            
            # Calculate industry performance
            industry_performance = calculate_industry_performance(performance_df)
            
            # Display top 5 industries
            st.subheader("Top 5 Performing Industries")
            top_5_industries = industry_performance.head(5)
            
            # Create a styled DataFrame for top 5 industries
            top_5_display = top_5_industries.copy()
            top_5_display['Average Performance (%)'] = top_5_display['Average Performance (%)'].apply(lambda x: f"{x:+.2f}%")
            top_5_display['Average Volatility (%)'] = top_5_display['Average Volatility (%)'].apply(lambda x: f"{x:.2f}%")
            
            # Get ETFs for each industry
            industry_etfs = {}
            for industry in top_5_industries.index:
                industry_etfs[industry] = performance_df[performance_df['Industry'] == industry]['ETF'].tolist()
            
            # Add ETFs column
            top_5_display['ETFs'] = top_5_display.index.map(lambda x: ', '.join(industry_etfs[x]))
            
            # Reorder columns
            top_5_display = top_5_display[[
                'Average Performance (%)', 'Average Volatility (%)', 
                'Number of ETFs', 'ETFs'
            ]]
            
            st.dataframe(
                top_5_display.style.apply(
                    lambda x: ['background-color: #e6ffe6' if i % 2 == 0 else '' for i in range(len(x))],
                    axis=0
                ),
                use_container_width=True
            )
            
            # Create industry performance chart for top 5
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_5_industries.index,
                y=top_5_industries['Average Performance (%)'],
                name='Industry Performance',
                marker_color='green',
                text=top_5_industries['Average Performance (%)'].apply(lambda x: f"{x:+.2f}%"),
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f"Top 5 Industry Performance ({period})",
                xaxis_title="Industry",
                yaxis_title="Average Performance (%)",
                xaxis_tickangle=45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display top 10 performers
            st.subheader("Top 10 Performing ETFs")
            top_10 = performance_df.head(10)
            
            # Create a styled DataFrame for top 10
            top_10_display = top_10.copy()
            top_10_display['Performance (%)'] = top_10_display['Performance (%)'].apply(lambda x: f"{x:+.2f}%")
            top_10_display['Volatility (%)'] = top_10_display['Volatility (%)'].apply(lambda x: f"{x:.2f}%")
            top_10_display['Current Price'] = top_10_display['Current Price'].apply(lambda x: f"${x:.2f}")
            top_10_display['Start Price'] = top_10_display['Start Price'].apply(lambda x: f"${x:.2f}")
            
            # Select and reorder columns
            top_10_display = top_10_display[[
                'ETF', 'Industry', 'Performance (%)', 
                'Volatility (%)', 'Current Price', 'Start Price'
            ]]
            
            st.dataframe(
                top_10_display.style.apply(
                    lambda x: ['background-color: #e6ffe6' if i % 2 == 0 else '' for i in range(len(x))],
                    axis=0
                ),
                use_container_width=True
            )
            
            # Display bottom 10 performers
            st.subheader("Bottom 10 Performing ETFs")
            bottom_10 = performance_df.tail(10)
            
            # Create a styled DataFrame for bottom 10
            bottom_10_display = bottom_10.copy()
            bottom_10_display['Performance (%)'] = bottom_10_display['Performance (%)'].apply(lambda x: f"{x:+.2f}%")
            bottom_10_display['Volatility (%)'] = bottom_10_display['Volatility (%)'].apply(lambda x: f"{x:.2f}%")
            bottom_10_display['Current Price'] = bottom_10_display['Current Price'].apply(lambda x: f"${x:.2f}")
            bottom_10_display['Start Price'] = bottom_10_display['Start Price'].apply(lambda x: f"${x:.2f}")
            
            # Select and reorder columns
            bottom_10_display = bottom_10_display[[
                'ETF', 'Industry', 'Performance (%)', 
                'Volatility (%)', 'Current Price', 'Start Price'
            ]]
            
            st.dataframe(
                bottom_10_display.style.apply(
                    lambda x: ['background-color: #ffe6e6' if i % 2 == 0 else '' for i in range(len(x))],
                    axis=0
                ),
                use_container_width=True
            )
            
            # Industry Performance Analysis
            st.subheader("Industry Performance Analysis")
            
            # Create industry performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=industry_performance.index,
                y=industry_performance['Average Performance (%)'],
                name='Industry Performance',
                marker_color='blue',
                text=industry_performance['Average Performance (%)'].apply(lambda x: f"{x:+.2f}%"),
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f"Industry Performance Comparison ({period})",
                xaxis_title="Industry",
                yaxis_title="Average Performance (%)",
                xaxis_tickangle=45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display industry performance table
            industry_display = industry_performance.copy()
            industry_display['Average Performance (%)'] = industry_display['Average Performance (%)'].apply(lambda x: f"{x:+.2f}%")
            industry_display['Average Volatility (%)'] = industry_display['Average Volatility (%)'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(
                industry_display.style.apply(
                    lambda x: ['background-color: #e6f3ff' if i % 2 == 0 else '' for i in range(len(x))],
                    axis=0
                ),
                use_container_width=True
            )
            
            # Create ETF performance comparison chart
            fig = go.Figure()
            
            # Add top 10 performers
            fig.add_trace(go.Bar(
                x=top_10.apply(lambda x: f"{x['ETF']} ({x['Industry']})", axis=1),
                y=top_10['Performance (%)'],
                name='Top 10 Performers',
                marker_color='green',
                text=top_10['Performance (%)'].apply(lambda x: f"{x:+.2f}%"),
                textposition='auto',
            ))
            
            # Add bottom 10 performers
            fig.add_trace(go.Bar(
                x=bottom_10.apply(lambda x: f"{x['ETF']} ({x['Industry']})", axis=1),
                y=bottom_10['Performance (%)'],
                name='Bottom 10 Performers',
                marker_color='red',
                text=bottom_10['Performance (%)'].apply(lambda x: f"{x:+.2f}%"),
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f"ETF Performance Comparison ({period})",
                xaxis_title="ETF (Industry)",
                yaxis_title="Performance (%)",
                barmode='group',
                xaxis_tickangle=45,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics
            st.subheader("Performance Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Best Performing ETF",
                    f"{top_10.iloc[0]['ETF']} ({top_10.iloc[0]['Industry']})",
                    f"{top_10.iloc[0]['Performance (%)']:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Worst Performing ETF",
                    f"{bottom_10.iloc[0]['ETF']} ({bottom_10.iloc[0]['Industry']})",
                    f"{bottom_10.iloc[0]['Performance (%)']:+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Best Performing Industry",
                    f"{industry_performance.index[0]}",
                    f"{industry_performance['Average Performance (%)'].iloc[0]:+.2f}%"
                )

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
    
    # Add ETF Performance Analysis at the top
    display_etf_performance()
    
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
        # print(df)
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
        #lstm_metrics = ml_predictor.train_lstm()
        
        if rf_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Random Forest R² Score", f"{rf_metrics['r2']:.3f}")
            # with col2:
            #     st.metric("LSTM R² Score", f"{lstm_metrics['r2']:.3f}")
            
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