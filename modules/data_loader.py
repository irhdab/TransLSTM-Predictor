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
        Handle outliers using IQR method.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        print("Handling outliers...")
        df = df.copy()
        initial_rows = len(df)
        
        # Apply IQR method to close prices
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        df_filtered = df[(df['close'] >= lower_bound) & (df['close'] <= upper_bound)]
        final_rows = len(df_filtered)
        
        print(f"Outliers handled: {initial_rows - final_rows} rows removed")
        return df_filtered

    def extract_features(self, df):
        """
        Extract OHLCV features from DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV columns
            
        Returns:
            np.array: Array of shape (N, 5) with OHLCV features
        """
        print("Extracting features...")
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize volume to prevent large values from dominating
        features[:, 4] = features[:, 4] / np.max(features[:, 4])
        
        print(f"✓ Features extracted: {features.shape}")
        return features

    def normalize_data(self, features):
        """
        Normalize features using MinMaxScaler.
        
        Args:
            features (np.array): Array of features to normalize
            
        Returns:
            tuple: (normalized_features, scaler)
        """
        print("Normalizing data...")
        self.scaler = MinMaxScaler()
        normalized_features = self.scaler.fit_transform(features)
        
        # Verify normalization
        print(f"Normalization range: [{np.min(normalized_features):.3f}, {np.max(normalized_features):.3f}]")
        print("✓ Data normalization completed")
        return normalized_features, self.scaler

    def create_sequences(self, data, original_dates):
        """
        Create sequences for time series prediction.
        
        Args:
            data (np.array): Normalized data array of shape (N, features)
            original_dates (pd.Series): Original dates corresponding to the data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, test_dates)
        """
        print("Creating sequences...")
        seq_length = self.config.SEQ_LENGTH
        sequences = []
        targets = []
        
        # Create sequences
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length, 3])  # Close price is index 3
            
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split into train/test
        split_idx = int(len(sequences) * (1 - self.config.TEST_SPLIT_RATIO))
        
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        y_train = targets[:split_idx]
        y_test = targets[split_idx:]
        
        # Get corresponding test dates
        test_dates = original_dates[seq_length + split_idx:].reset_index(drop=True)
        
        print(f"Sequences created:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - y_test: {y_test.shape}")
        print(f"  - test_dates: {test_dates.shape}")
        
        return X_train, X_test, y_train, y_test, test_dates

    def get_scaler(self):
        """
        Get the fitted scaler for inverse transformation.
        
        Returns:
            MinMaxScaler: Fitted scaler object
        """
        return self.scaler
