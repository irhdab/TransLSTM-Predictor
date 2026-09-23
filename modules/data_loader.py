import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging
from types import ModuleType


logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, csv_path: str, config: ModuleType):
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
        # Aligned post-indicator state (fixes pre/post-dropna misalignment)
        self.data_with_indicators: pd.DataFrame | None = None
        self.feature_dates: pd.Series | None = None
        self.feature_close: np.ndarray | None = None

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading raw data")
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = [col.strip().lower().replace('_', ' ') for col in df.columns]
            # Normalize common variants: 'adj close', 'adj_close' -> drop (use 'close')
            for adj in ('adj close', 'adjclose'):
                if adj in df.columns:
                    logger.info("Dropping redundant column '%s' (using 'close')", adj)
                    df = df.drop(columns=[adj])
            logger.info("Data loaded successfully: %s", df.shape)
            return df
        except Exception as e:
            logger.error("Error loading data: %s", e)
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity and structure.

        Args:
            df (pd.DataFrame): DataFrame to validate

        Returns:
            bool: True if data is valid, False otherwise
        """
        logger.info("Validating data")
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error("Missing required columns: %s", missing_columns)
            return False

        # Minimum length check (need at least SEQ + FUTURE + indicator warmup + margin)
        min_required = self.config.SEQ_LENGTH + self.config.FUTURE_DAYS + 30
        if len(df) < min_required:
            logger.error(
                "Not enough rows: got %d, need at least %d (SEQ=%d + FUTURE=%d + warmup)",
                len(df), min_required, self.config.SEQ_LENGTH, self.config.FUTURE_DAYS,
            )
            return False

        # Check for non-positive prices (splits/errors) - fail fast
        for col in ('open', 'high', 'low', 'close'):
            if (df[col] <= 0).any():
                logger.error("Non-positive values in '%s' detected (check splits/bad ticks)", col)
                return False

        # Check for negative volume
        if (df['volume'] < 0).any():
            logger.error("Negative volume values detected")
            return False

        # Check logical consistency (open included - was missing before)
        if (df['high'] < df['low']).any():
            logger.error("Inconsistent high/low values detected")
            return False

        if (df['high'] < df['close']).any():
            logger.error("Inconsistent high/close values detected")
            return False

        if (df['low'] > df['close']).any():
            logger.error("Inconsistent low/close values detected")
            return False

        if ((df['open'] > df['high']) | (df['open'] < df['low'])).any():
            logger.error("Inconsistent open/high/low values detected")
            return False

        # Check duplicate dates
        if df['date'].astype(str).duplicated().any():
            logger.error("Duplicate dates detected")
            return False

        # Missing values: report but don't hard-fail on price gaps -
        # forward-fill small gaps instead of rejecting real-world CSVs.
        # (Full NaN rejection broke on normal Yahoo Finance downloads.)
        n_missing = int(df[required_columns].isnull().sum().sum())
        if n_missing > 0:
            logger.warning("Missing values detected: %d (will forward-fill)", n_missing)
            # Too much missing -> fail
            if n_missing > len(df) * 0.05:
                logger.error("Too many missing values (>5%%), aborting")
                return False

        logger.info("Data validation passed")
        return True

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and sort date column.

        Args:
            df (pd.DataFrame): DataFrame with date column

        Returns:
            pd.DataFrame: DataFrame with parsed and sorted dates
        """
        logger.info("Parsing dates")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        # Forward-fill small price gaps so real CSVs don't get rejected
        price_cols = [c for c in ('open', 'high', 'low', 'close', 'volume') if c in df.columns]
        df[price_cols] = df[price_cols].ffill()
        df = df.dropna(subset=price_cols).reset_index(drop=True)
        logger.info("Dates parsed and sorted (%d rows)", len(df))
        return df

    def handle_outliers(self, df: pd.DataFrame, fit_end_idx: int | None = None) -> pd.DataFrame:
        """
        Handle outliers using IQR method on price returns by clipping.

        Leakage fix: bounds are fitted on the TRAIN prefix only
        (``df.iloc[:fit_end_idx]``). If ``fit_end_idx`` is None, the first
        60% of the series is used as a causal proxy so future crashes/rallies
        never leak into the bounds. The bounds are then *applied* to the full
        series to preserve time-series continuity (no rows removed).

        OHLC fix: only clipping ``close`` left ``high < close`` violations.
        After adjusting ``close`` we repair ``high``/``low`` so that
        ``high >= max(open, close)`` and ``low <= min(open, close)``.

        Args:
            df: DataFrame with OHLC columns.
            fit_end_idx: Row index (exclusive) defining the train prefix used
                to fit the IQR bounds. None -> first 60% of rows.
        """
        logger.info("Handling outliers using price returns (train-only bounds, clipping)")
        df = df.copy()

        n = len(df)
        if fit_end_idx is None:
            fit_end_idx = max(30, int(n * 0.6))
        fit_end_idx = max(10, min(int(fit_end_idx), n))

        # Calculate daily returns on the FIT prefix only
        fit_returns = df['close'].iloc[:fit_end_idx].pct_change()

        # Apply IQR method to returns (5/95 percentiles = lenient for stocks)
        Q1 = fit_returns.quantile(0.05)
        Q3 = fit_returns.quantile(0.95)
        IQR = Q3 - Q1
        # Degenerate case (flat prices): skip clipping
        if not np.isfinite(IQR) or IQR == 0:
            logger.warning("IQR is degenerate, skipping outlier clipping")
            return df

        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR

        # Clip returns on the FULL series using train-fitted bounds
        returns = df['close'].pct_change()
        outlier_mask = (returns < lower_bound) | (returns > upper_bound)
        returns_clipped = returns.clip(lower=lower_bound, upper=upper_bound)

        # Reconstruct the close price ONLY for the outlier days
        df.loc[outlier_mask, 'close'] = df['close'].shift(1)[outlier_mask] * (1 + returns_clipped[outlier_mask])

        # Repair OHLC consistency broken by close-only adjustment
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))

        logger.info(
            "Outliers handled: %s values clipped (bounds fitted on first %d rows, "
            "OHLC repaired, no rows removed)",
            int(outlier_mask.sum()), fit_end_idx,
        )
        return df

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate technical indicators and extract features.

        Uses Wilder's smoothing for RSI (was simple SMA before).
        Stores date/close aligned to the post-dropna frame in
        ``self.feature_dates`` / ``self.feature_close`` so downstream
        sequence/date/price alignment is exact.
        """
        logger.info("Calculating technical indicators and extracting features")
        df = df.copy()

        # Moving Averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()

        # RSI (Wilder's smoothing via EMA, alpha=1/14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        # Flat market (avg_loss == 0 and avg_gain == 0) -> RSI 50, not NaN
        flat = (avg_gain == 0) & (avg_loss == 0)
        df.loc[flat, 'rsi'] = 50.0
        # Pure gains -> 100
        df.loc[(avg_loss == 0) & (avg_gain > 0), 'rsi'] = 100.0
        df['rsi'] = df['rsi'].fillna(50.0)

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
        # Aligned post-dropna dates/prices - THE fix for the 21-day misalignment
        self.feature_dates = df['date'].reset_index(drop=True)
        self.feature_close = df['close'].values

        missing = [c for c in self.config.FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns after engineering: {missing}")
        features = df[self.config.FEATURE_COLS].values.astype(np.float64)
        if not np.all(np.isfinite(features)):
            raise ValueError("Non-finite values in feature matrix after engineering")
        logger.info("Features extracted: %s", features.shape)
        return features

    def normalize_data(
        self,
        features: np.ndarray,
        targets: np.ndarray | None = None,
        train_end: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, RobustScaler, RobustScaler | None]:
        """
        Normalize features and targets separately to prevent scale mismatch and leakage.

        Args:
            features (np.array): Feature matrix.
            targets (np.array): Target matrix.
            train_end (int, optional): The index where training data ends.
                                       If None, uses config.TEST_SPLIT_RATIO.
        """
        logger.info("Normalizing data (dual scaler system with leakage prevention)")

        if train_end is None:
            seq_length = self.config.SEQ_LENGTH
            future_days = self.config.FUTURE_DAYS
            total_samples = len(features) - seq_length - future_days + 1
            train_end = int(total_samples * (1 - self.config.TEST_SPLIT_RATIO)) + seq_length

        train_end = max(1, min(int(train_end), len(features)))
        # Guard against degenerate (constant) features
        if len(np.unique(features[:train_end].reshape(-1))) < 2:
            raise ValueError("Training features are constant, cannot fit scaler")

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
            target_train_cutoff = max(1, train_end - self.config.SEQ_LENGTH)
            target_train_cutoff = min(target_train_cutoff, len(targets))
            self.target_scaler.fit(targets[:target_train_cutoff])
            norm_targets = self.target_scaler.transform(targets)

        logger.info("Normalization completed (fitted up to index %s)", train_end)
        return norm_features, norm_targets, self.scaler, self.target_scaler

    def create_sequences(self, data: np.ndarray, original_dates: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Create multi-step sequences. Returns RAW targets for normalization.

        Date fix: ``original_dates`` from the pre-indicator frame is
        misaligned by the indicator warmup (~21 rows). If lengths mismatch,
        the aligned ``self.feature_dates`` (post-dropna) is used instead.

        Target date semantics: ``dates[i]`` is the date of the LAST bar in
        ``sequences[i]``; the target covers the NEXT ``FUTURE_DAYS`` bars.
        """
        logger.info("Creating multi-step sequences (raw targets)")
        seq_length = self.config.SEQ_LENGTH
        future_days = self.config.FUTURE_DAYS

        if self.data_with_indicators is None:
            raise ValueError("Call extract_features() before create_sequences()")
        if len(data) < seq_length + future_days:
            raise ValueError(
                f"Not enough data for sequences: got {len(data)}, "
                f"need >= {seq_length + future_days}"
            )

        # Resolve aligned dates
        dates = original_dates
        if dates is None or len(dates) != len(data):
            if self.feature_dates is not None and len(self.feature_dates) == len(data):
                logger.warning(
                    "Date length mismatch (%s vs %d), using aligned post-indicator dates",
                    None if dates is None else len(dates), len(data),
                )
                dates = self.feature_dates
            else:
                raise ValueError(
                    f"Date/feature length mismatch: dates={None if dates is None else len(dates)}, "
                    f"features={len(data)}. Pass post-indicator dates."
                )
        dates = pd.Series(dates).reset_index(drop=True)

        sequences = []
        targets = []

        target_data = self.data_with_indicators['close'].values
        if self.config.PREDICT_RETURNS:
            target_data = self.data_with_indicators['daily_return'].values

        for i in range(len(data) - seq_length - future_days + 1):
            sequences.append(data[i:i + seq_length])
            targets.append(target_data[i + seq_length:i + seq_length + future_days])

        seq_dates = dates.iloc[seq_length - 1: seq_length - 1 + len(sequences)].reset_index(drop=True)
        return np.array(sequences), np.array(targets), seq_dates

    def get_aligned_close(self) -> np.ndarray:
        """Return post-indicator close prices aligned to features."""
        if self.feature_close is None:
            raise ValueError("Call extract_features() first")
        return self.feature_close

    def get_aligned_dates(self) -> pd.Series:
        """Return post-indicator dates aligned to features."""
        if self.feature_dates is None:
            raise ValueError("Call extract_features() first")
        return self.feature_dates

    def get_scaler(self) -> RobustScaler | None:
        """
        Get the fitted scaler for inverse transformation.

        Returns:
            RobustScaler: Fitted scaler object
        """
        return self.scaler
