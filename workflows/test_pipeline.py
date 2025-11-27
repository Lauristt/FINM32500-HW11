import unittest
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.append(ROOT)

from feature_engineering import run_feature_engineering
from train_model import train_models_with_5folds
from signal_generator import generate_signals_for_all_models
from backtest import run_backtest_for_signals

class TestMLPipeline(unittest.TestCase):

    def test_feature_generation(self):
        run_feature_engineering()
        self.assertTrue(os.path.exists("processed_data.csv"))
        df = pd.read_csv("processed_data.csv")
        self.assertFalse(df.empty)

    def test_model_training(self):
        train_models_with_5folds()
        self.assertTrue(os.path.exists("best_model.pkl"))
        self.assertTrue(os.path.exists("test_data_with_actuals.csv"))

    def test_no_lookahead_bias(self):
        run_feature_engineering()
        df = pd.read_csv("processed_data.csv")

        bad_cols = []
        for col in df.columns:
            if col not in ["direction", "future_return", "close"]:
                if df[col].shift(1).equals(df["close"].shift(-1)):
                    bad_cols.append(col)
        self.assertEqual(len(bad_cols), 0, f"Look-ahead features detected: {bad_cols}")

    def test_signal_generation(self):
        generate_signals_for_all_models()
        found = any(f.startswith("trading_signals_") for f in os.listdir("."))
        self.assertTrue(found)

    def test_backtest(self):
        df = run_backtest_for_signals()
        self.assertFalse(df.empty)
        self.assertIn("Model", df.columns)

if __name__ == "__main__":
    unittest.main()
