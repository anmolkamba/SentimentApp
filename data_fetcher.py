import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import praw
import requests
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.setup_reddit()
        self.setup_newsapi()
    
    def setup_reddit(self):
        """Setup Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
    
    def setup_newsapi(self):
        """Setup NewsAPI client"""
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    
    def get_stock_data(self, symbol, period='1y'):
        """
        Fetch historical stock data using yfinance
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_social_media_data(self, symbol, days=7):
        """
        Fetch social media data from Reddit and news data from NewsAPI
        """
        try:
            # Get Reddit data
            reddit_data = self.get_reddit_data(symbol)
            
            # Get news data
            news_data = self.get_news_data(symbol)
            
            # Combine data
            all_data = []
            
            # Process Reddit data
            if reddit_data:
                for post in reddit_data:
                    all_data.append({
                        'text': post['title'] + ' ' + post['text'],
                        'created_at': post['created_at'],
                        'user': post['author'],
                        'followers': post['score'],  # Using post score as a proxy for influence
                        'source': 'reddit',
                        'likes': post['score'],
                        'comments': post['num_comments']
                    })
            
            # Process news data
            if news_data:
                for article in news_data:
                    all_data.append({
                        'text': article['title'] + ' ' + article['description'],
                        'created_at': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                        'user': article['source']['name'],
                        'followers': 0,  # News sources don't have follower counts
                        'source': 'news',
                        'likes': 0,
                        'comments': 0
                    })
            
            return pd.DataFrame(all_data)
            
        except Exception as e:
            print(f"Error fetching social media data for {symbol}: {str(e)}")
            return None
    
    def get_reddit_data(self, symbol):
        """Fetch data from relevant Reddit subreddits"""
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        posts = []
        
        try:
            for subreddit in subreddits:
                subreddit = self.reddit.subreddit(subreddit)
                # Search for posts about the symbol
                for post in subreddit.search(f"${symbol}", limit=20):
                    posts.append({
                        'title': post.title,
                        'text': post.selftext,
                        'created_at': datetime.fromtimestamp(post.created_utc),
                        'author': post.author.name if post.author else '[deleted]',
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'subreddit': subreddit.display_name
                    })
        except Exception as e:
            print(f"Error fetching Reddit data: {str(e)}")
        
        return posts
    
    def get_news_data(self, symbol):
        """Fetch news articles from NewsAPI"""
        try:
            # Get news from the last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            news = self.newsapi.get_everything(
                q=f"{symbol} stock",
                from_param=from_date,
                language='en',
                sort_by='relevancy'
            )
            
            return news.get('articles', [])
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
            return None
    
    def get_technical_indicators(self, df):
        """
        Calculate technical indicators for the given dataframe
        """
        if df is None or df.empty:
            return None
            
        # Calculate basic technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line 