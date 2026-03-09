import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import warnings
import os
import sys

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import config

class DataProcessor:
    def __init__(self, csv_path, config):
        """
        Initialize the DataProcessor with CSV path and configuration.
        
        Args:
            csv_path (str): Path to the CSV file containing stock data
            config (module): Configuration module with data processing parameters
        """
        self.csv_path = csv_path
        self.config = config
        self.scaler = None
        self.target_scaler = None
        self.data = None

    def load_raw_data(self):
        """
        Load raw data from CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        print("Loading raw data...")
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = [col.strip().lower() for col in df.columns]
            print(f"✓ Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def validate_data(self, df):
        """
        Validate data integrity and structure.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        print("Validating data...")
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for NaN values
        if df.isnull().sum().sum() > 0:
            print("Missing values detected:")
            print(df.isnull().sum())
            return False
            
        # Check for negative volume
        if (df['volume'] < 0).any():
            print("Negative volume values detected")
            return False
            
        # Check logical consistency
        if (df['high'] < df['low']).any():
            print("Inconsistent high/low values detected")
            return False
            
        if (df['high'] < df['close']).any():
            print("Inconsistent high/close values detected")
            return False
            
        if (df['low'] > df['close']).any():
            print("Inconsistent low/close values detected")
            return False
            
        print("✓ Data validation passed")
        return True

    def parse_dates(self, df):
        """
        Parse and sort date column.
        
        Args:
            df (pd.DataFrame): DataFrame with date column
            
        Returns:
            pd.DataFrame: DataFrame with parsed and sorted dates
        """
        print("Parsing dates...")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print("✓ Dates parsed and sorted")
        return df

    def handle_outliers(self, df):
        """
        Handle outliers using IQR method on price returns by clipping (prevents time-series gaps).
        """
        print("Handling outliers using price returns (Clipping)...")
        df = df.copy()
        
        # Calculate daily returns
        returns = df['close'].pct_change()
        
        # Apply IQR method to returns
        Q1 = returns.quantile(0.05) # Be more lenient for stock data
        Q3 = returns.quantile(0.95)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        
        # Identify outlier mask (ignore the first NaN return)
        outlier_mask = (returns < lower_bound) | (returns > upper_bound)
        
        # Clip the returns
        returns_clipped = returns.clip(lower=lower_bound, upper=upper_bound)
        
        # Reconstruct the close price ONLY for the outlier days using the clipped return
        # Prevents continuous drift across the entire time series
        df.loc[outlier_mask, 'close'] = df['close'].shift(1)[outlier_mask] * (1 + returns_clipped[outlier_mask])
        
        print(f"Outliers handled: {outlier_mask.sum()} values clipped to boundaries (no rows removed, time-series continuity preserved).")
        return df

    def extract_features(self, df):
        """
        Calculate technical indicators and extract features.
        """
        print("Calculating technical indicators and extracting features...")
        df = df.copy()
        
        # Moving Averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        # Bollinger Bands
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['bollinger_h'] = df['ma_20'] + (df['std_20'] * 2)
        df['bollinger_l'] = df['ma_20'] - (df['std_20'] * 2)
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Daily Returns for target
        df['daily_return'] = df['close'].pct_change()
        
        # Drop NaNs created by indicators
        df = df.dropna().reset_index(drop=True)
        self.data_with_indicators = df
        
        features = df[self.config.FEATURE_COLS].values
        print(f"✓ Features extracted: {features.shape}")
        return features

    def normalize_data(self, features, targets=None, train_end=None):
        """
        Normalize features and targets separately to prevent scale mismatch and leakage.
        
        Args:
            features (np.array): Feature matrix.
            targets (np.array): Target matrix.
            train_end (int, optional): The index where training data ends. 
                                      If None, uses config.TEST_SPLIT_RATIO.
        """
        print("Normalizing data (Dual Scaler System with Leakage Prevention)...")
        
        if train_end is None:
            seq_length = self.config.SEQ_LENGTH
            future_days = self.config.FUTURE_DAYS
            total_samples = len(features) - seq_length - future_days + 1
            train_end = int(total_samples * (1 - self.config.TEST_SPLIT_RATIO)) + seq_length
        
        # 1. Feature Scaling
        self.scaler = RobustScaler()
        self.scaler.fit(features[:train_end])
        norm_features = self.scaler.transform(features)
        
        # 2. Target Scaling
        norm_targets = None
        if targets is not None:
            self.target_scaler = RobustScaler()
            # Targets are aligned with sequences starting at seq_length
            # So targets[0] matches sequence ending at seq_length - 1
            # If features[:train_end] is used, targets up to index (train_end - seq_length) are valid for training
            target_train_cutoff = max(0, train_end - self.config.SEQ_LENGTH)
            self.target_scaler.fit(targets[:target_train_cutoff])
            norm_targets = self.target_scaler.transform(targets)
            
        print(f"✓ Normalization completed (Fitted up to index {train_end})")
        return norm_features, norm_targets, self.scaler, self.target_scaler

    def create_sequences(self, data, original_dates):
        """
        Create multi-step sequences. Returns RAW targets for normalization.
        """
        print("Creating multi-step sequences (Raw Targets)...")
        seq_length = self.config.SEQ_LENGTH
        future_days = self.config.FUTURE_DAYS
        
        sequences = []
        targets = []
        
        target_data = self.data_with_indicators['close'].values
        if self.config.PREDICT_RETURNS:
            target_data = self.data_with_indicators['daily_return'].values
            
        for i in range(len(data) - seq_length - future_days + 1):
            sequences.append(data[i:i + seq_length])
            targets.append(target_data[i + seq_length:i + seq_length + future_days])
            
        return np.array(sequences), np.array(targets), original_dates.iloc[seq_length - 1 : seq_length - 1 + len(sequences)].reset_index(drop=True)

    def get_scaler(self):
        """
        Get the fitted scaler for inverse transformation.
        
        Returns:
            MinMaxScaler: Fitted scaler object
        """
        return self.scaler
