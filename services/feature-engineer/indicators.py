"""
Technical Indicators Calculation Module
Vectorized implementation using NumPy and Pandas for performance
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Array of prices
        period: RSI period (default 14)
        
    Returns:
        Array of RSI values (0-100)
    """
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)
    
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return np.full(len(prices), 100.0)
    
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        rs = up / (down + 1e-10)
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi


def calculate_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Array of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (macd, signal_line, histogram)
    """
    prices_series = pd.Series(prices)
    
    ema_fast = prices_series.ewm(span=fast, adjust=False).mean().values
    ema_slow = prices_series.ewm(span=slow, adjust=False).mean().values
    
    macd = ema_fast - ema_slow
    signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
        
    Returns:
        Array of ATR values
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = np.nan  # First value undefined
    
    # Calculate ATR using EMA
    atr = pd.Series(true_range).ewm(span=period, adjust=False).mean().values
    
    return atr


def calculate_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Array of prices
        period: Moving average period
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band, bandwidth)
    """
    prices_series = pd.Series(prices)
    
    middle_band = prices_series.rolling(window=period).mean().values
    std = prices_series.rolling(window=period).std().values
    
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    bandwidth = (upper_band - lower_band) / (middle_band + 1e-10)
    
    return upper_band, middle_band, lower_band, bandwidth


def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        k_period: %K period
        d_period: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    # Calculate %K
    lowest_low = pd.Series(low).rolling(window=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period).max().values
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    # Calculate %D (smoothed %K)
    d = pd.Series(k).rolling(window=d_period).mean().values
    
    return k, d


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Array of prices
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Array of prices
        period: SMA period
        
    Returns:
        Array of SMA values
    """
    return pd.Series(prices).rolling(window=period).mean().values


def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        close: Array of close prices
        volume: Array of volumes
        
    Returns:
        Array of OBV values
    """
    obv = np.zeros(len(close))
    obv[0] = volume[0]
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv


def calculate_vpt(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Calculate Volume Price Trend (VPT).
    
    Args:
        close: Array of close prices
        volume: Array of volumes
        
    Returns:
        Array of VPT values
    """
    vpt = np.zeros(len(close))
    
    for i in range(1, len(close)):
        vpt[i] = vpt[i-1] + volume[i] * ((close[i] - close[i-1]) / (close[i-1] + 1e-10))
    
    return vpt


def calculate_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        returns: Array of returns
        window: Rolling window size
        
    Returns:
        Array of volatility values
    """
    return pd.Series(returns).rolling(window=window).std().values


def calculate_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """
    Calculate returns.
    
    Args:
        prices: Array of prices
        periods: Number of periods for return calculation
        
    Returns:
        Array of returns
    """
    returns = np.zeros_like(prices)
    returns[periods:] = (prices[periods:] - prices[:-periods]) / (prices[:-periods] + 1e-10)
    return returns


class TechnicalIndicatorCalculator:
    """
    Calculate all technical indicators for a DataFrame.
    
    Usage:
        calculator = TechnicalIndicatorCalculator()
        df_with_indicators = calculator.calculate_all(df)
    """
    
    def __init__(self):
        """Initialize calculator."""
        pass
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
            
        Returns:
            DataFrame with additional indicator columns
        """
        if df.empty or len(df) < 30:
            return df
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Extract arrays
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Momentum Indicators
        df['rsi_14'] = calculate_rsi(close, period=14)
        
        macd, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Volatility Indicators
        df['atr_14'] = calculate_atr(high, low, close, period=14)
        
        bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, num_std=2.0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_width
        
        returns = calculate_returns(close, periods=1)
        df['volatility_20'] = calculate_volatility(returns, window=20)
        
        # Trend Indicators
        df['ema_12'] = calculate_ema(close, period=12)
        df['ema_26'] = calculate_ema(close, period=26)
        df['sma_20'] = calculate_sma(close, period=20)
        
        # Volume Indicators
        df['obv'] = calculate_obv(close, volume)
        df['vpt'] = calculate_vpt(close, volume)
        df['volume_sma'] = calculate_sma(volume, period=20)
        
        # Additional features
        df['returns'] = returns
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
        """
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in df.columns:
                return False
        
        return True


class FeatureNormalizer:
    """
    Normalize features using Z-score normalization.
    Stores normalization parameters for later use in inference.
    """
    
    def __init__(self):
        """Initialize normalizer."""
        self.mean_dict = {}
        self.std_dict = {}
        self.normalized = False
    
    def fit(self, df: pd.DataFrame, columns: Optional[list] = None) -> 'FeatureNormalizer':
        """
        Fit normalizer to data (calculate mean and std).
        
        Args:
            df: DataFrame with features
            columns: List of columns to normalize (None = all numeric columns)
            
        Returns:
            Self for method chaining
        """
        if columns is None:
            # Auto-detect numeric columns (exclude time and price columns)
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            columns = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        for col in columns:
            if col in df.columns:
                self.mean_dict[col] = df[col].mean()
                self.std_dict[col] = df[col].std()
        
        self.normalized = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        if not self.normalized:
            raise ValueError("Normalizer must be fitted before transform")
        
        df = df.copy()
        
        for col in self.mean_dict.keys():
            if col in df.columns:
                mean = self.mean_dict[col]
                std = self.std_dict[col]
                
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and normalize
            columns: Columns to normalize
            
        Returns:
            Normalized DataFrame
        """
        self.fit(df, columns)
        return self.transform(df)
    
    def get_params(self) -> dict:
        """
        Get normalization parameters.
        
        Returns:
            Dictionary with mean and std for each feature
        """
        return {
            'mean': self.mean_dict,
            'std': self.std_dict
        }
