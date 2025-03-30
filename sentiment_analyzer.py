import pandas as pd
from textblob import TextBlob
import numpy as np
from collections import defaultdict

class SentimentAnalyzer:
    def __init__(self, social_data):
        self.social_data = social_data
        self.sentiment_scores = None
    
    def analyze_sentiment(self):
        """Analyze sentiment of social media posts"""
        if self.social_data is None or self.social_data.empty:
            return None
            
        # Initialize sentiment scores
        self.sentiment_scores = defaultdict(list)
        
        # Analyze each post
        for _, row in self.social_data.iterrows():
            # Get sentiment polarity and subjectivity
            blob = TextBlob(row['text'])
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Store sentiment scores
            self.sentiment_scores['polarity'].append(polarity)
            self.sentiment_scores['subjectivity'].append(subjectivity)
            
            # Add sentiment labels
            if polarity > 0.1:
                self.sentiment_scores['label'].append('Positive')
            elif polarity < -0.1:
                self.sentiment_scores['label'].append('Negative')
            else:
                self.sentiment_scores['label'].append('Neutral')
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(self.sentiment_scores)
        
        # Add timestamp from original data
        sentiment_df['timestamp'] = self.social_data['created_at']
        
        return sentiment_df
    
    def get_sentiment_summary(self):
        """Get summary statistics of sentiment analysis"""
        if self.sentiment_scores is None:
            return None
            
        summary = {
            'average_polarity': np.mean(self.sentiment_scores['polarity']),
            'average_subjectivity': np.mean(self.sentiment_scores['subjectivity']),
            'sentiment_distribution': {
                'positive': len([x for x in self.sentiment_scores['label'] if x == 'Positive']),
                'negative': len([x for x in self.sentiment_scores['label'] if x == 'Negative']),
                'neutral': len([x for x in self.sentiment_scores['label'] if x == 'Neutral'])
            }
        }
        
        return summary
    
    def get_sentiment_trend(self, window_size=7):
        """Get sentiment trend over time"""
        if self.sentiment_scores is None:
            return None
            
        # Create DataFrame with sentiment scores
        sentiment_df = pd.DataFrame(self.sentiment_scores)
        sentiment_df['timestamp'] = self.social_data['created_at']
        
        # Sort by timestamp
        sentiment_df = sentiment_df.sort_values('timestamp')
        
        # Calculate rolling average of polarity
        sentiment_df['rolling_polarity'] = sentiment_df['polarity'].rolling(window=window_size).mean()
        
        return sentiment_df
    
    def get_impact_score(self):
        """Calculate sentiment impact score considering user influence"""
        if self.social_data is None or self.social_data.empty:
            return None
            
        # Calculate weighted sentiment score based on follower count
        total_followers = self.social_data['followers'].sum()
        weighted_sentiment = 0
        
        for _, row in self.social_data.iterrows():
            blob = TextBlob(row['text'])
            weight = row['followers'] / total_followers
            weighted_sentiment += blob.sentiment.polarity * weight
        
        return weighted_sentiment
    
    def get_keywords(self):
        """Extract important keywords from social media posts"""
        if self.social_data is None or self.social_data.empty:
            return None
            
        # Combine all text
        all_text = ' '.join(self.social_data['text'])
        
        # Create TextBlob object
        blob = TextBlob(all_text)
        
        # Get noun phrases and words
        noun_phrases = blob.noun_phrases
        words = blob.words
        
        # Count frequencies
        phrase_freq = defaultdict(int)
        word_freq = defaultdict(int)
        
        for phrase in noun_phrases:
            phrase_freq[phrase] += 1
            
        for word in words:
            if len(word) > 3:  # Filter out short words
                word_freq[word] += 1
        
        # Sort by frequency
        top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'top_phrases': top_phrases,
            'top_words': top_words
        } 