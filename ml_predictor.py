import pandas as pd
import numpy as np
import sys 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

class MLPredictor:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        self.model = None
        self.lstm_model = None
    
    def prepare_data(self):
        """Prepare data for machine learning models"""
        if self.data is None or self.data.empty:
            return None, None
            
        # Create features
        features = pd.DataFrame()
        
        # Technical indicators
        features['RSI'] = self.data['RSI']
        features['MACD'] = self.data['MACD']
        features['BB_Upper'] = self.data['BB_Upper']
        features['BB_Lower'] = self.data['BB_Lower']
        
        # Price-based features
        features['Price_Change'] = self.data['Close'].pct_change()
        features['Volume_Change'] = self.data['Volume'].pct_change()
        features['High_Low_Range'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        
        # Moving averages
        features['SMA_20'] = self.data['SMA_20']
        features['SMA_50'] = self.data['SMA_50']
        
        # Remove NaN values
        features = features.dropna()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create target variable (next day's closing price)
        target = self.data['Close'].shift(-1).dropna()
        
        # Align features and target
        scaled_features = scaled_features[:-1]
        
        return scaled_features, target
    
    def train_random_forest(self):
        """Train Random Forest model"""
        X, y = self.prepare_data()
        if X is None or y is None:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {'mse': mse, 'r2': r2}
    
    def prepare_lstm_data(self, sequence_length=10):
        """Prepare data for LSTM model"""
        X, y = self.prepare_data()
        if X is None or y is None:
            return None, None
            
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:(i + sequence_length)])
            y_sequences.append(y.iloc[i + sequence_length])
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_lstm(self, sequence_length=10):
        """Train LSTM model"""
        X, y = self.prepare_lstm_data(sequence_length)
        if X is None or y is None:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compile model
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Evaluate model
        y_pred = self.lstm_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {'mse': mse, 'r2': r2, 'history': history.history}
    
    def predict_price(self, days=5):
        """Predict future prices using both models"""
        if self.model is None or self.lstm_model is None:
            return None
            
        # Get last sequence of data
        X, _ = self.prepare_data()
        if X is None:
            return None
            
        # Random Forest prediction
        rf_predictions = []
        current_features = X[-1]
        
        for _ in range(days):
            rf_pred = self.model.predict([current_features])[0]
            rf_predictions.append(rf_pred)
            # Update features for next prediction
            current_features = np.roll(current_features, -1)
            current_features[-1] = rf_pred
        
        # LSTM prediction
        X_lstm, _ = self.prepare_lstm_data()
        if X_lstm is not None:
            last_sequence = X_lstm[-1]
            lstm_predictions = []
            
            for _ in range(days):
                lstm_pred = self.lstm_model.predict([last_sequence])[0][0]
                lstm_predictions.append(lstm_pred)
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = lstm_pred
        else:
            lstm_predictions = None
        
        return {
            'random_forest': rf_predictions,
            'lstm': lstm_predictions
        } 