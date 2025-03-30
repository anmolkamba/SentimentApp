import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class TechnicalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.indicators = {}
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None or self.data.empty:
            return None
            
        # Calculate SMA
        sma20 = SMAIndicator(close=self.data['Close'], window=20)
        sma50 = SMAIndicator(close=self.data['Close'], window=50)
        self.data['SMA_20'] = sma20.sma_indicator()
        self.data['SMA_50'] = sma50.sma_indicator()
        
        # Calculate RSI
        rsi = RSIIndicator(close=self.data['Close'])
        self.data['RSI'] = rsi.rsi()
        
        # Calculate MACD
        macd = MACD(close=self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        
        # Calculate Bollinger Bands
        bb = BollingerBands(close=self.data['Close'])
        self.data['BB_Upper'] = bb.bollinger_hband()
        self.data['BB_Lower'] = bb.bollinger_lband()
        
        return self.data
    
    def generate_signals(self):
        """Generate trading signals based on technical indicators"""
        signals = pd.DataFrame(index=self.data.index)
        
        # RSI signals
        signals['RSI_Signal'] = np.where(self.data['RSI'] > 70, -1,
                                       np.where(self.data['RSI'] < 30, 1, 0))
        
        # MACD signals
        signals['MACD_Signal'] = np.where(self.data['MACD'] > self.data['MACD_Signal'], 1,
                                        np.where(self.data['MACD'] < self.data['MACD_Signal'], -1, 0))
        
        # Moving Average signals
        signals['MA_Signal'] = np.where(self.data['SMA_20'] > self.data['SMA_50'], 1,
                                      np.where(self.data['SMA_20'] < self.data['SMA_50'], -1, 0))
        
        # Bollinger Bands signals
        signals['BB_Signal'] = np.where(self.data['Close'] > self.data['BB_Upper'], -1,
                                      np.where(self.data['Close'] < self.data['BB_Lower'], 1, 0))
        
        # Combine signals
        signals['Combined_Signal'] = (signals['RSI_Signal'] + signals['MACD_Signal'] + 
                                    signals['MA_Signal'] + signals['BB_Signal']) / 4
        
        return signals
    
    def predict_price(self, days=5):
        """Predict future price based on technical indicators"""
        if self.data is None or self.data.empty:
            return None
            
        # Calculate price momentum
        self.data['Price_Momentum'] = self.data['Close'].pct_change(periods=5)
        
        # Calculate volatility
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        
        # Simple prediction based on current trend and momentum
        last_price = self.data['Close'].iloc[-1]
        last_momentum = self.data['Price_Momentum'].iloc[-1]
        last_volatility = self.data['Volatility'].iloc[-1]
        
        predictions = []
        current_price = last_price
        
        for _ in range(days):
            # Adjust price based on momentum and volatility
            price_change = (last_momentum * current_price) + (np.random.normal(0, last_volatility))
            current_price = current_price * (1 + price_change)
            predictions.append(current_price)
        
        return predictions 