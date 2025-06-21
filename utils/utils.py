from dataclasses import dataclass
import pandas as pd
import numpy as np 
from typing import Optional, List

## configs
@dataclass
class DatasetConfig:
    window_size: int = 60
    horizon: int = 10
    features: Optional[List[str]] = None
    target_type: str = "classification"  # "classification" or "regression"
    scaling_method: str = "standard"  # "standard", "robust", "minmax", "none"
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42


@dataclass
class ModelConfig:
    """Configuration for the LSTM model"""
    input_size: int = 12
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True
    use_residual: bool = True
    use_layer_norm: bool = True
    output_size: int = 1
    task_type: str = "classification"  # "classification", "regression", "multi_class"
    num_classes: int = 2  # for multi-class classification
    gradient_clip: float = 1.0
    weight_decay: float = 1e-4


class TechnicalIndicators:
    """Technical indicators calculator with proper validation"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate input data quality"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        invalid_rows = (
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        )
        
        if invalid_rows.any():
            logging.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC data")
            df = df[~invalid_rows].copy()
        
        return df
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Price position within day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])  
        df['price_position'] = df['price_position'].fillna(0.5)  # Handle division by zero
        
        # Volatility measures
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages"""
        df = df.copy()
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            df[f'close_sma_{period}_ratio'] = df['close'] / df[f'SMA_{period}']
        
        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'close_ema_{period}_ratio'] = df['close'] / df[f'EMA_{period}']
        
        # Volume moving averages
        df['Vol_SMA_10'] = df['volume'].rolling(window=10).mean()
        df['Vol_SMA_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['Vol_SMA_20']
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        df = df.copy()
        
            # RSI with proper calculation
        def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(0)
        
        df['RSI_14'] = calculate_rsi(df['close'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Stochastic Oscillator
        df['lowest_low_14'] = df['low'].rolling(window=14).min()
        df['highest_high_14'] = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - df['lowest_low_14']) / (
            df['highest_high_14'] - df['lowest_low_14']
        )
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        df = df.copy()
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands
        bb_mid = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = bb_mid + 2 * bb_std
        df['BB_lower'] = bb_mid - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_mid
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        df = df.copy()
        
        # On-Balance Volume
        df['price_change'] = df['close'].diff()
        df['OBV'] = 0
        df.loc[df['price_change'] > 0, 'OBV'] = df['volume']
        df.loc[df['price_change'] < 0, 'OBV'] = -df['volume']
        df['OBV'] = df['OBV'].cumsum()
        
        # Volume Price Trend
        df['VPT'] = (df['volume'] * df['returns']).cumsum()
        
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        df['MFI'] = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return df
    
    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators"""
        df = cls.validate_data(df)
        df = cls.add_price_features(df)
        df = cls.add_moving_averages(df)
        df = cls.add_momentum_indicators(df)
        df = cls.add_volatility_indicators(df)
        df = cls.add_volume_indicators(df)
        
        # Clean up temporary columns
        temp_columns = ['price_change', 'lowest_low_14', 'highest_high_14']
        df = df.drop(columns=[col for col in temp_columns if col in df.columns])
        
        return df
