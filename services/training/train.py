"""
Training Service - Full Production Implementation
Daily model retraining with walk-forward validation
LSTM + XGBoost ensemble on Chainlink price data
"""

import os
import sys
import time
import signal
import pickle
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from apscheduler.schedulers.background import BackgroundScheduler



from db_client import create_db_client_from_env
from logger import setup_logger_from_env
import constants

logger = setup_logger_from_env('training')


# ============================================================================
# LSTM Model Definition
# ============================================================================

class LSTMFeatureExtractor(nn.Module):
    """LSTM for temporal feature extraction."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
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
        
        self.fc = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        features = self.relu(self.fc(last_hidden))
        return features


class SequenceDataset(Dataset):
    """Dataset for LSTM sequences."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================================
# Training Service
# ============================================================================

class TrainingService:
    """
    Full production training service.
    Trains LSTM + XGBoost ensemble daily.
    """
    
    def __init__(
        self,
        schedule_cron: str = constants.TRAINING_SCHEDULE_CRON,
        lookback_months: int = constants.TRAINING_LOOKBACK_MONTHS
    ):
        """Initialize training service."""
        self.schedule_cron = schedule_cron
        self.lookback_months = lookback_months
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")
        
        # Database
        self.db = create_db_client_from_env()
        
        # Scheduler
        self.scheduler = BackgroundScheduler()
        self.shutdown_requested = False
        
        # Model paths
        self.model_dir = constants.MODEL_ACTIVE_PATH
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(constants.MODEL_ARCHIVE_PATH, exist_ok=True)
        
        logger.info("Training Service initialized")
    
    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load historical Chainlink data for training."""
        try:
            lookback_days = self.lookback_months * 30
            
            query = f"""
                SELECT 
                    o.time, o.open, o.high, o.low, o.close, o.volume,
                    t.rsi_14, t.macd, t.macd_hist, t.atr_14, t.stoch_k,
                    t.ema_12, t.ema_26, t.volatility_20, t.bb_width
                FROM {constants.TABLE_OHLCV_RAW} o
                LEFT JOIN {constants.TABLE_TECHNICAL_INDICATORS} t ON o.time = t.time
                WHERE o.time > NOW() - INTERVAL '{lookback_days} days'
                    AND o.source = 'chainlink'
                ORDER BY o.time ASC
            """
            
            df = pd.DataFrame(self.db.fetch_all(query))
            
            if df.empty:
                logger.error("No Chainlink training data found")
                return None
            
            # Remove rows with missing indicators
            indicator_cols = ['rsi_14', 'macd', 'atr_14', 'ema_12', 'volatility_20']
            df = df.dropna(subset=indicator_cols)
            
            if len(df) < 1000:
                logger.error(f"Insufficient training data: {len(df)} rows (need >= 1000)")
                return None
            
            logger.info(f"Loaded {len(df)} training samples from Chainlink (ground truth)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None
    
    def create_labels(self, df: pd.DataFrame, horizon_minutes: int = 15) -> pd.DataFrame:
        """Create binary labels for direction prediction."""
        df = df.copy()
        df['future_close'] = df['close'].shift(-horizon_minutes)
        df['label'] = (df['future_close'] >= df['close']).astype(int)
        df = df[:-horizon_minutes]
        
        logger.info(f"Created labels: {df['label'].sum()} UP, {len(df) - df['label'].sum()} DOWN")
        return df
    
    def create_sequences(self, prices: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        """Create overlapping sequences for LSTM."""
        sequences = []
        
        for i in range(len(prices) - sequence_length):
            seq = prices[i:i + sequence_length]
            seq_normalized = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)
            sequences.append(seq_normalized)
        
        sequences = np.array(sequences)
        sequences = sequences.reshape(-1, sequence_length, 1)
        return sequences
    
    def train_lstm(self, sequences: np.ndarray, labels: np.ndarray, val_split: float = 0.2) -> Tuple[LSTMFeatureExtractor, Dict[str, Any]]:
        """Train LSTM feature extractor."""
        logger.info("Starting LSTM training...")
        
        split_idx = int(len(sequences) * (1 - val_split))
        train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        train_dataset = SequenceDataset(train_seq, train_labels)
        val_dataset = SequenceDataset(val_seq, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=constants.LSTM_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=constants.LSTM_BATCH_SIZE, shuffle=False)
        
        model = LSTMFeatureExtractor(
            input_size=constants.LSTM_INPUT_SIZE,
            hidden_size=constants.LSTM_HIDDEN_SIZE,
            num_layers=constants.LSTM_NUM_LAYERS,
            dropout=constants.LSTM_DROPOUT
        ).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=constants.LSTM_LEARNING_RATE)
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(constants.LSTM_EPOCHS):
            model.train()
            train_loss = 0.0
            
            for batch_seq, batch_labels in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                features = model(batch_seq)
                logits = features.mean(dim=1)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_seq, batch_labels in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    features = model(batch_seq)
                    logits = features.mean(dim=1)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(batch_labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_true, val_preds)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{constants.LSTM_EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= constants.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"LSTM training complete. Best val_loss: {best_val_loss:.4f}")
        return model, history
    
    def extract_lstm_features(self, model: LSTMFeatureExtractor, sequences: np.ndarray) -> np.ndarray:
        """Extract LSTM features for entire dataset."""
        model.eval()
        features_list = []
        
        dataset = SequenceDataset(sequences, np.zeros(len(sequences)))
        loader = DataLoader(dataset, batch_size=constants.LSTM_BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for batch_seq, _ in loader:
                batch_seq = batch_seq.to(self.device)
                features = model(batch_seq)
                features_list.append(features.cpu().numpy())
        
        all_features = np.vstack(features_list)
        logger.info(f"Extracted LSTM features: shape {all_features.shape}")
        return all_features
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[xgb.Booster, Dict[str, Any]]:
        """Train XGBoost classifier."""
        logger.info("Starting XGBoost training...")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': constants.XGB_MAX_DEPTH,
            'learning_rate': constants.XGB_LEARNING_RATE,
            'n_estimators': constants.XGB_N_ESTIMATORS,
            'min_child_weight': constants.XGB_MIN_CHILD_WEIGHT,
            'subsample': constants.XGB_SUBSAMPLE,
            'colsample_bytree': constants.XGB_COLSAMPLE_BYTREE,
            'gamma': constants.XGB_GAMMA,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        booster = xgb.train(params, dtrain, num_boost_round=constants.XGB_N_ESTIMATORS, evals=evals, early_stopping_rounds=10, verbose_eval=10)
        
        y_pred_proba = booster.predict(dval)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_val, y_pred_proba)
        }
        
        logger.info(f"XGBoost metrics: {metrics}")
        return booster, metrics
    
    def save_models(self, lstm_model: LSTMFeatureExtractor, xgb_model: xgb.Booster, version: str):
        """Save trained models to disk."""
        lstm_path = os.path.join(self.model_dir, constants.LSTM_MODEL_FILENAME)
        torch.save(lstm_model.state_dict(), lstm_path)
        logger.info(f"Saved LSTM model to {lstm_path}")
        
        xgb_path = os.path.join(self.model_dir, constants.XGB_MODEL_FILENAME)
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        logger.info(f"Saved XGBoost model to {xgb_path}")
    
    def update_model_registry(self, metrics: Dict[str, Any], version: str):
        """Update model registry in database."""
        try:
            data = {
                'model_name': constants.MODEL_A,
                'version': version,
                'status': 'active',
                'lstm_path': os.path.join(self.model_dir, constants.LSTM_MODEL_FILENAME),
                'xgb_path': os.path.join(self.model_dir, constants.XGB_MODEL_FILENAME),
                'accuracy': metrics.get('accuracy'),
                'precision_score': metrics.get('precision'),
                'recall_score': metrics.get('recall'),
                'f1_score': metrics.get('f1'),
                'auc_roc': metrics.get('auc_roc'),
                'trained_at': datetime.utcnow(),
                'deployed_at': datetime.utcnow()
            }
            
            self.db.insert(constants.TABLE_MODEL_REGISTRY, data, on_conflict="DO NOTHING")
            logger.info(f"Updated model registry: {version}")
            
        except Exception as e:
            logger.error(f"Failed to update model registry: {e}")
    
    def train_models(self):
        """Main training pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            df = self.load_training_data()
            if df is None:
                logger.error("Training aborted: no data")
                return
            
            df = self.create_labels(df, horizon_minutes=15)
            
            train_size = int(len(df) * constants.TRAIN_SPLIT)
            val_size = int(len(df) * constants.VALIDATION_SPLIT)
            
            train_df = df[:train_size]
            val_df = df[train_size:train_size + val_size]
            test_df = df[train_size + val_size:]
            
            logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            
            train_sequences = self.create_sequences(train_df['close'].values)
            val_sequences = self.create_sequences(val_df['close'].values)
            
            train_labels = train_df['label'].values[constants.LSTM_SEQUENCE_LENGTH:]
            val_labels = val_df['label'].values[constants.LSTM_SEQUENCE_LENGTH:]
            
            lstm_model, lstm_history = self.train_lstm(train_sequences, train_labels)
            
            all_sequences = self.create_sequences(df['close'].values)
            lstm_features = self.extract_lstm_features(lstm_model, all_sequences)
            
            indicator_cols = ['rsi_14', 'macd', 'macd_hist', 'atr_14', 'stoch_k', 'ema_12', 'ema_26', 'volatility_20', 'bb_width']
            tech_features = df[indicator_cols].iloc[constants.LSTM_SEQUENCE_LENGTH:].fillna(0).values
            combined_features = np.hstack([lstm_features, tech_features])
            combined_labels = df['label'].values[constants.LSTM_SEQUENCE_LENGTH:]
            
            train_end = len(train_df) - constants.LSTM_SEQUENCE_LENGTH
            val_end = train_end + len(val_df)
            
            X_train = combined_features[:train_end]
            y_train = combined_labels[:train_end]
            X_val = combined_features[train_end:val_end]
            y_val = combined_labels[train_end:val_end]
            
            xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_val, y_val)
            
            version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.save_models(lstm_model, xgb_model, version)
            self.update_model_registry(xgb_metrics, version)
            
            duration = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"Training Complete in {duration:.2f}s")
            logger.info(f"Version: {version}")
            logger.info(f"Accuracy: {xgb_metrics['accuracy']:.4f}")
            logger.info(f"AUC-ROC: {xgb_metrics['auc_roc']:.4f}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
    
    def start(self):
        """Start training service with scheduler."""
        logger.info(f"Starting Training Service (schedule: {self.schedule_cron})")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        cron_parts = self.schedule_cron.split()
        
        self.scheduler.add_job(
            self.train_models,
            'cron',
            minute=cron_parts[0] if len(cron_parts) > 0 else '0',
            hour=cron_parts[1] if len(cron_parts) > 1 else '2',
            day=cron_parts[2] if len(cron_parts) > 2 else '*',
            month=cron_parts[3] if len(cron_parts) > 3 else '*',
            day_of_week=cron_parts[4] if len(cron_parts) > 4 else '*',
            id='training_job'
        )
        
        self.scheduler.start()
        logger.info("Training scheduler started")
        
        logger.info("Running initial training...")
        self.train_models()
        
        try:
            while not self.shutdown_requested:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.shutdown_requested = True
    
    def shutdown(self):
        logger.info("Shutting down Training Service...")
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        
        if hasattr(self, 'db'):
            self.db.close()
        
        logger.info("Training Service shutdown complete")


def main():
    """Entry point."""
    logger.info("=" * 80)
    logger.info("Training Service Starting - FULL PRODUCTION")
    logger.info("=" * 80)
    logger.info("Training on Chainlink BTC/USD data (Polymarket ground truth)")
    logger.info("=" * 80)
    
    try:
        service = TrainingService()
        service.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
