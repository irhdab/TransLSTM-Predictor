import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
        Handle outliers using IQR method on price returns.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        print("Handling outliers using price returns...")
        df = df.copy()
        
        # Calculate daily returns
        returns = df['close'].pct_change()
        
        # Apply IQR method to returns
        Q1 = returns.quantile(0.05) # Be more lenient for stock data
        Q3 = returns.quantile(0.95)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        
        # Keep rows where returns are within bounds (first row is NaN, keep it)
        mask = (returns >= lower_bound) & (returns <= upper_bound)
        mask.iloc[0] = True 
        
        df_filtered = df[mask].reset_index(drop=True)
        print(f"Outliers handled: {len(df) - len(df_filtered)} rows removed")
        return df_filtered

    def extract_features(self, df):
        """
        Calculate technical indicators and extract features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV columns
            
        Returns:
            np.array: Array of features defined in config
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
        
        # Drop NaNs created by indicators
        df = df.dropna().reset_index(drop=True)
        self.data_with_indicators = df # Store for date alignment if needed
        
        features = df[self.config.FEATURE_COLS].values
        print(f"✓ Features extracted: {features.shape}")
        return features

    def normalize_data(self, features):
        """
        Normalize features using MinMaxScaler.
        """
        print("Normalizing data...")
        self.scaler = MinMaxScaler()
        normalized_features = self.scaler.fit_transform(features)
        print("✓ Data normalization completed")
        return normalized_features, self.scaler

    def create_sequences(self, data, original_dates):
        """
        Create multi-step sequences for time series prediction.
        Target is a window of FUTURE_DAYS.
        """
        print("Creating multi-step sequences...")
        seq_length = self.config.SEQ_LENGTH
        future_days = self.config.FUTURE_DAYS
        
        sequences = []
        targets = []
        
        # We need enough data to have both a sequence and a future window
        for i in range(len(data) - seq_length - future_days + 1):
            sequences.append(data[i:i + seq_length])
            # Target is the 'close' price (index CLOSE_COL_INDEX) for the next FUTURE_DAYS
            targets.append(data[i + seq_length:i + seq_length + future_days, self.config.CLOSE_COL_INDEX])
            
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Align dates with the end of the input sequence
        # The test_dates should correspond to the point where we make the prediction
        # For a given sequence [i : i+seq_length], the "current" date is original_dates[i + seq_length - 1]
        aligned_dates = original_dates.iloc[seq_length - 1 : seq_length - 1 + len(sequences)].reset_index(drop=True)
        
        # Split into train/test
        split_idx = int(len(sequences) * (1 - self.config.TEST_SPLIT_RATIO))
        
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        y_train = targets[:split_idx]
        y_test = targets[split_idx:]
        test_dates = aligned_dates[split_idx:].reset_index(drop=True)
        
        print(f"Sequences created:")
        print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, test_dates

    def get_scaler(self):
        """
        Get the fitted scaler for inverse transformation.
        
        Returns:
            MinMaxScaler: Fitted scaler object
        """
        return self.scaler
