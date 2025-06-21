import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, Dict, Union
import warnings
import logging

from utils import DatasetConfig, TechnicalIndicators

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
        df = TechnicalIndicators.add_all_indicators(df)
        df = self._handle_missing_values(df)
        feature_columns = self._select_features(df)
        self.feature_names = feature_columns
        
        # Prepare features and targets
        features = df[feature_columns].values.astype(np.float32)
        targets = self._create_targets(df)

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

        df = df.fillna(method='ffill').fillna(method='bfill')
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            self.logger.warning(f"Dropped {initial_len - len(df)} rows due to missing values")
        
        return df.reset_index(drop=True)
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features for training"""
        if self.config.features:
            return self.config.features
        
        base_features = ['open', 'high', 'low', 'close', 'volume']
        technical_features = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20',
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            'close_sma_20_ratio', 'close_ema_20_ratio',
            'RSI_14', 'MACD', 'MACD_signal', 'MACD_histogram',
            'ATR_14', 'BB_width', 'BB_position',
            'stoch_k', 'stoch_d', 'MFI',
            'volume_ratio', 'price_position', 'hl_ratio'
        ]
        
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

