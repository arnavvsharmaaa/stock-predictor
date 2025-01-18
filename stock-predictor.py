import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List

class StockPredictor:
    def __init__(self, symbol: str, prediction_days: int = 60):
       
        self.symbol = symbol
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='stock_predictions.log'
        )

    def fetch_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            stock_data = yf.download(self.symbol, start=start_date, end=end_date)
            logging.info(f"Successfully fetched data for {self.symbol}")
            return stock_data
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Extract closing prices and convert to numpy array
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X = []
        y = []
        
        for i in range(self.prediction_days, len(scaled_data)):
            X.append(scaled_data[i-self.prediction_days:i, 0])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape data for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        logging.info("Model built and compiled successfully")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   epochs: int = 25, batch_size: int = 32,
                   validation_split: float = 0.1) -> tf.keras.callbacks.History:
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        logging.info("Model training completed")
        return history

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
       
        predictions = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform([y_test])
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test_actual.T, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test_actual.T, predictions)),
            'mae': mean_absolute_error(y_test_actual.T, predictions),
            'r2': r2_score(y_test_actual.T, predictions)
        }
        
        logging.info(f"Model evaluation metrics: {metrics}")
        return metrics

    def predict_future(self, days: int = 30) -> List[float]:
       
        # Get the last 'prediction_days' of data
        data = yf.download(self.symbol, 
                          start=(datetime.now() - timedelta(days=self.prediction_days+days)).strftime('%Y-%m-%d'),
                          end=datetime.now().strftime('%Y-%m-%d'))
        
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1, 1))
        
        predictions = []
        current_batch = scaled_data[-self.prediction_days:]
        
        for _ in range(days):
            # Prepare current batch for prediction
            current_batch_reshaped = current_batch.reshape(1, self.prediction_days, 1)
            
            # Get prediction
            prediction = self.model.predict(current_batch_reshaped)
            
            # Add prediction to list
            predictions.append(prediction[0, 0])
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[1:], prediction)
            
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        logging.info(f"Generated {days} days of future predictions")
        return predictions.flatten().tolist()

    def plot_predictions(self, actual_prices: np.ndarray, predicted_prices: np.ndarray,
                        title: str = "Stock Price Prediction") -> None:
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual Prices')
        plt.plot(predicted_prices, label='Predicted Prices')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize predictor for netflix stock
    predictor = StockPredictor('NFLX')

    
    try:
        # Fetch historical data
        data = predictor.fetch_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(data)
        
        # Build and train model
        predictor.build_model((predictor.prediction_days, 1))
        history = predictor.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Make future predictions
        future_predictions = predictor.predict_future(days=30)
        print("\nPredicted prices for next 30 days:")
        for i, price in enumerate(future_predictions, 1):
            print(f"Day {i}: ${price:.2f}")
            
    except Exception as e:
        logging.error(f"Prediction process failed: {str(e)}")
