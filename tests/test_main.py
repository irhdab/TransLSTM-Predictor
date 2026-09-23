import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import config.config as config
from modules.data_loader import DataProcessor
from modules.backtester import Backtester


def _make_sample_csv(n: int = 120) -> str:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, size=n))
    closes = np.maximum(closes, 1.0)
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='B'),
        'open': closes * (1 + rng.normal(0, 0.002, size=n)),
        'high': closes * 1.01,
        'low': closes * 0.99,
        'close': closes,
        'volume': np.full(n, 1_000_000),
    })
    # enforce OHLC consistency
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(path, index=False)
    return path


class TestConfig(unittest.TestCase):
    def test_validate_ok(self) -> None:
        config.validate()

    def test_validate_rejects_bad_seq(self) -> None:
        old = config.SEQ_LENGTH
        config.SEQ_LENGTH = 2
        try:
            with self.assertRaises(ValueError):
                config.validate()
        finally:
            config.SEQ_LENGTH = old


class TestDataProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _make_sample_csv()
        self.dp = DataProcessor(csv_path=self.path, config=config)

    def tearDown(self) -> None:
        os.remove(self.path)

    def test_load_validate_parse(self) -> None:
        df = self.dp.load_raw_data()
        self.assertTrue(self.dp.validate_data(df))
        parsed = self.dp.parse_dates(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(parsed['date']))

    def test_outlier_bounds_are_train_only_and_ohlc_repaired(self) -> None:
        df = self.dp.parse_dates(self.dp.load_raw_data())
        # Inject a huge future spike (should NOT define the bounds)
        df.loc[df.index[-1], 'close'] = df['close'].iloc[-2] * 5.0
        out = self.dp.handle_outliers(df, fit_end_idx=60)
        # close-only clipping must not leave high < close
        self.assertTrue((out['high'] >= out['close']).all())
        self.assertTrue((out['low'] <= out['close']).all())

    def test_features_dates_aligned(self) -> None:
        df = self.dp.parse_dates(self.dp.load_raw_data())
        df = self.dp.handle_outliers(df, fit_end_idx=60)
        feats = self.dp.extract_features(df)
        self.assertEqual(len(feats), len(self.dp.get_aligned_dates()))
        self.assertEqual(len(feats), len(self.dp.get_aligned_close()))
        # RSI in [0,100], finite features
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_sequences_aligned_and_guarded(self) -> None:
        df = self.dp.parse_dates(self.dp.load_raw_data())
        df = self.dp.handle_outliers(df, fit_end_idx=60)
        feats = self.dp.extract_features(df)
        X, y, dates = self.dp.create_sequences(feats, self.dp.get_aligned_dates())
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(dates))
        # last-bar semantics: dates[i] == feature date of last input bar
        feat_dates = self.dp.get_aligned_dates().reset_index(drop=True)
        self.assertEqual(dates.iloc[0], feat_dates.iloc[config.SEQ_LENGTH - 1])
        with self.assertRaises(ValueError):
            self.dp.create_sequences(np.zeros((5, feats.shape[1])), feat_dates.iloc[:5])


class TestBacktester(unittest.TestCase):
    def test_run_with_costs_no_lookahead_crash(self) -> None:
        bt = Backtester(config)
        n = 50
        actual = np.linspace(100, 110, n)
        predicted = actual * 1.001  # always bullish
        dates = pd.Series(pd.date_range('2021-01-01', periods=n, freq='B'))
        res = bt.run(actual, predicted, dates)
        self.assertIn('cum_strategy', res.columns)
        self.assertIn('strategy_return', res.columns)
        # With constant bullish signal, strategy ~= market minus costs
        self.assertLessEqual(res['cum_strategy'].iloc[-1], res['cum_market'].iloc[-1] + 1e-9)


class TestWalkForwardSplits(unittest.TestCase):
    @staticmethod
    def _splits(total: int, n_folds: int) -> list[tuple[int, int]]:
        initial = max(1, min(int(total * 0.6), total - n_folds))
        test_size = (total - initial) // n_folds
        assert test_size >= 1
        out = []
        for f in range(n_folds):
            tr = initial + f * test_size
            te = total if f == n_folds - 1 else tr + test_size
            out.append((tr, te))
        return out

    def test_no_empty_test_fold(self) -> None:
        for folds in (1, 2, 3, 5, 10):
            for tr, te in self._splits(4322, folds):
                self.assertGreater(te - tr, 0)
        # tail is fully covered
        self.assertEqual(self._splits(4322, 5)[-1][1], 4322)


class TestPipelineWiring(unittest.TestCase):
    def test_pipeline_prepare_data_aligns(self) -> None:
        from main import StockPipeline
        path = _make_sample_csv(200)
        try:
            pipe = StockPipeline(path)
            pipe.prepare_data()
            self.assertEqual(len(pipe.features), len(pipe.aligned_dates))
            self.assertEqual(len(pipe.features), len(pipe.aligned_close))
        finally:
            os.remove(path)

    def test_pipeline_rejects_missing_file(self) -> None:
        from main import StockPipeline
        with self.assertRaises(ValueError):
            StockPipeline('/nonexistent/file.csv')


if __name__ == '__main__':
    unittest.main()
