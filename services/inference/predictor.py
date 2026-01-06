"""
Prediction Module
LSTM feature extraction + XGBoost classification
"""

import os
import sys
from typing import Tuple, Optional
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared import constants
from shared.logger import setup_logger_from_env

logger = setup_logger_from_env('predictor')


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM model for temporal feature extraction.
    Outputs latent features instead of direct predictions.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM feature extractor.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection to fixed size (32 features)
        self.fc = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Latent features of shape (batch, 32)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # Shape: (batch, hidden_size)
        
        # Project to output size
        features = self.relu(self.fc(last_hidden))
        
        return features


class Predictor:
    """
    Combined LSTM + XGBoost predictor.
    """
    
    def __init__(self, model_path: str = constants.MODEL_ACTIVE_PATH):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Predictor using device: {self.device}")
        
        # Models
        self.lstm_model: Optional[LSTMFeatureExtractor] = None
        self.xgb_model: Optional[xgb.Booster] = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load LSTM and XGBoost models from disk."""
        try:
            # Load LSTM
            lstm_path = os.path.join(self.model_path, constants.LSTM_MODEL_FILENAME)
            
            if os.path.exists(lstm_path):
                self.lstm_model = LSTMFeatureExtractor(
                    input_size=constants.LSTM_INPUT_SIZE,
                    hidden_size=constants.LSTM_HIDDEN_SIZE,
                    num_layers=constants.LSTM_NUM_LAYERS,
                    dropout=constants.LSTM_DROPOUT
                )
                
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
                self.lstm_model = self.lstm_model.to(self.device)
                self.lstm_model.eval()
                
                logger.info(f"LSTM model loaded from {lstm_path}")
            else:
                logger.warning(f"LSTM model not found at {lstm_path}, using random init")
                self.lstm_model = LSTMFeatureExtractor()
                self.lstm_model = self.lstm_model.to(self.device)
                self.lstm_model.eval()
            
            # Load XGBoost
            xgb_path = os.path.join(self.model_path, constants.XGB_MODEL_FILENAME)
            
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                logger.info(f"XGBoost model loaded from {xgb_path}")
            else:
                logger.warning(f"XGBoost model not found at {xgb_path}")
                # Create dummy model for testing
                self.xgb_model = None
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_lstm_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract LSTM latent features from price sequence.
        
        Args:
            prices: Array of prices (length = sequence_length)
            
        Returns:
            Latent features array of shape (32,)
        """
        try:
            # Normalize prices
            prices_normalized = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
            
            # Reshape for LSTM: (batch=1, sequence_length, features=1)
            prices_tensor = torch.FloatTensor(prices_normalized).unsqueeze(0).unsqueeze(-1)
            prices_tensor = prices_tensor.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                features = self.lstm_model(prices_tensor)
            
            # Convert to numpy
            features_np = features.cpu().numpy().flatten()
            
            return features_np
            
        except Exception as e:
            logger.error(f"LSTM feature extraction failed: {e}")
            # Return zeros as fallback
            return np.zeros(32)
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Generate prediction from combined features.
        
        Args:
            features: Combined feature vector (LSTM + technical indicators)
            
        Returns:
            Tuple of (direction, confidence)
            direction: 'UP' or 'DOWN'
            confidence: Probability score (0.5 to 1.0)
        """
        try:
            if self.xgb_model is None:
                # Fallback: random prediction
                prob_up = np.random.rand()
                direction = constants.DIRECTION_UP if prob_up > 0.5 else constants.DIRECTION_DOWN
                confidence = max(prob_up, 1 - prob_up)
                
                logger.warning("Using random prediction (no XGBoost model)")
                return direction, confidence
            
            # Reshape for XGBoost
            features_reshaped = features.reshape(1, -1)
            dmatrix = xgb.DMatrix(features_reshaped)
            
            # Predict probability
            prob_up = self.xgb_model.predict(dmatrix)[0]
            
            # Determine direction
            direction = constants.DIRECTION_UP if prob_up >= 0.5 else constants.DIRECTION_DOWN
            
            # Confidence is distance from 0.5
            confidence = max(prob_up, 1 - prob_up)
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback
            return constants.DIRECTION_UP, 0.5


def prepare_inference_features(
    df: pd.DataFrame,
    predictor: Predictor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare combined features for inference.
    
    Args:
        df: DataFrame with last 60 rows of OHLCV + indicators
        predictor: Predictor instance with LSTM model
        
    Returns:
        Tuple of (combined_features, lstm_features)
    """
    # Extract price sequence for LSTM
    prices = df['close'].values
    
    # Get LSTM features
    lstm_features = predictor.extract_lstm_features(prices)
    
    # Technical indicators (most recent row)
    indicator_cols = [
        'rsi_14', 'macd', 'macd_hist', 'atr_14', 'stoch_k',
        'ema_12', 'ema_26', 'volatility_20', 'bb_width'
    ]
    
    # Get available indicators
    available_indicators = [col for col in indicator_cols if col in df.columns]
    
    if available_indicators:
        tech_features = df[available_indicators].iloc[-1].fillna(0).values
    else:
        tech_features = np.zeros(len(indicator_cols))
    
    # Combine: LSTM (32) + technical indicators
    combined_features = np.concatenate([lstm_features, tech_features])
    
    return combined_features, lstm_features
