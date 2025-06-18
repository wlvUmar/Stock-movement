import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, Dict, Union
import warnings
from dataclasses import dataclass
import logging


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


class StockDataset(Dataset):
    """Improved stock dataset with proper ML practices"""
    
    def __init__(self, 
                 data: Union[str, pd.DataFrame], 
                 config: DatasetConfig,
                 mode: str = "train",
                 scaler: Optional[object] = None):
        """
        Initialize dataset
        
        Args:
            data: CSV path or DataFrame
            config: Dataset configuration
            mode: "train", "val", or "test"
            scaler: Pre-fitted scaler for validation/test sets
        """
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        if isinstance(data, str):
            df = pd.read_csv(data, parse_dates=["date"])
        else:
            df = data.copy()
        
        df = df.sort_values("date").reset_index(drop=True)
        
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Select features
        feature_columns = self._select_features(df)
        self.feature_names = feature_columns
        
        # Prepare features and targets
        features = df[feature_columns].values.astype(np.float32)
        targets = self._create_targets(df)
        
        # Apply scaling
        if scaler is None and mode == "train":
            self.scaler = self._fit_scaler(features)
        else:
            self.scaler = scaler
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Create sequences
        self.X, self.y = self._create_sequences(features, targets)
        
        self.logger.info(f"Created {mode} dataset with {len(self.X)} samples")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        # Forward fill first, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN values, drop those rows
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            self.logger.warning(f"Dropped {initial_len - len(df)} rows due to missing values")
        
        return df.reset_index(drop=True)
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features for training"""
        if self.config.features:
            return self.config.features
        
        # Default feature selection
        base_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Add key technical indicators
        technical_features = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20',
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            'close_sma_20_ratio', 'close_ema_20_ratio',
            'RSI_14', 'MACD', 'MACD_signal', 'MACD_histogram',
            'ATR_14', 'BB_width', 'BB_position',
            'stoch_k', 'stoch_d', 'MFI',
            'volume_ratio', 'price_position', 'hl_ratio'
        ]
        
        # Filter features that exist in the dataframe
        available_features = [col for col in base_features + technical_features 
                            if col in df.columns]
        
        return available_features
    
    def _create_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variables"""
        if self.config.target_type == "classification":
            # Binary classification: price goes up or down
            future_prices = df['close'].shift(-self.config.horizon)
            current_prices = df['close']
            targets = (future_prices > current_prices).astype(int)
        else:
            # Regression: predict future returns
            future_prices = df['close'].shift(-self.config.horizon)
            current_prices = df['close']
            targets = (future_prices - current_prices) / current_prices
        
        return targets.values
    
    def _fit_scaler(self, features: np.ndarray) -> object:
        """Fit scaler to training data"""
        if self.config.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_method == "robust":
            scaler = RobustScaler()
        elif self.config.scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            return None
        
        return scaler.fit(features)
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for time series prediction"""
        X, y = [], []
        
        # Ensure we have enough data
        max_idx = len(features) - self.config.window_size - self.config.horizon
        
        for i in range(max_idx):
            # Input sequence
            sequence = features[i:i + self.config.window_size]
            # Target (avoid future data leakage)
            target = targets[i + self.config.window_size + self.config.horizon - 1]
            
            if not np.isnan(target):  # Skip NaN targets
                X.append(sequence)
                y.append(target)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        if self.config.target_type == "classification":
            y = y.long()
        else:
            y = y.unsqueeze(-1)
        
        return X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_scaler(self):
        """Return the fitted scaler"""
        return self.scaler


def create_stock_datasets(csv_path: str, 
                         config: DatasetConfig) -> Tuple[StockDataset, StockDataset, StockDataset]:
    """Create train, validation, and test datasets with proper data splitting"""
    
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    total_len = len(df)
    test_size = int(total_len * config.test_split)
    val_size = int(total_len * config.validation_split)
    train_size = total_len - test_size - val_size
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    train_dataset = StockDataset(train_df, config, mode="train")
    val_dataset = StockDataset(val_df, config, mode="val", scaler=train_dataset.get_scaler())
    test_dataset = StockDataset(test_df, config, mode="test", scaler=train_dataset.get_scaler())
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset: StockDataset, 
                       val_dataset: StockDataset, 
                       test_dataset: StockDataset,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training"""
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = DatasetConfig(
        window_size=60,
        horizon=10,
        target_type="classification",
        scaling_method="robust",
        validation_split=0.2,
        test_split=0.1
    )
    
    train_ds, val_ds, test_ds = create_stock_datasets("../temp.csv", config)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=32
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Features: {len(train_ds.feature_names)}")
    print(f"Feature names: {train_ds.feature_names[:]}...")  # Show first 10