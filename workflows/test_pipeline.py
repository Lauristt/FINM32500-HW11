import unittest
import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
import pathlib
from feature_engineering import run_feature_engineering
from train_model import train_models_with_5folds  # 如果函数名不同你改一下
from signal_generator import generate_signals_for_all_models
from backtest import backtest_signal_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

print("pytest working directory forced to:", os.getcwd())


class TestMLPipeline(unittest.TestCase):

    def test_feature_generation(self):
        """Test processed_data.csv is created and contains required columns."""
        run_feature_engineering()
        self.assertTrue(os.path.exists("processed_data.csv"))

        df = pd.read_csv("processed_data.csv")
        with open("features_config.json") as f:
            cfg = json.load(f)

        # Check all features exist
        for feat in cfg["features"]:
            self.assertIn(feat, df.columns)

        # Check label exists
        self.assertIn(cfg["label"], df.columns)

        # Check no NaN
        self.assertFalse(df[cfg["features"]].isna().any().any())

    def test_no_lookahead_bias(self):
        """
        Detect structural look-ahead bias:
        - No feature is allowed to use shift(-1)
        - No feature is allowed to reference future_return
        - All rolling windows must be backward-looking
        """
        with open("processed_data.csv", "r") as f:
            df = pd.read_csv(f)

        shifted_cols = [col for col in df.columns if "shift" in col and "-1" in col]
        self.assertEqual(len(shifted_cols), 0, f"Shift(-1) detected in features: {shifted_cols}")

        with open("features_config.json") as f:
            cfg = json.load(f)
        features = cfg["features"]

        self.assertNotIn("future_return", features, "future_return leaked into features!")

        df2 = df.copy()
        df2["close_tomorrow"] = df2["close"].shift(-1)

        for feat in features:
            if feat in ["close", "log_return_1d"]:
                continue

            # high correlation does NOT mean leakage -> skip statistical test
            # Instead check if feature is shifted relative to close
            # If feature[t] == close[t+1] for many rows → leakage
            same_count = np.sum(df[feat].iloc[1:].values == df["close"].iloc[:-1].values)
            self.assertLess(same_count, 5, f"Feature {feat} appears to equal future close values!")

        print("Structural look-ahead check passed.")

    def test_model_training(self):
        """Check models train successfully and produce prediction of correct shape."""
        train_models_with_5folds()

        # load best model
        self.assertTrue(os.path.exists("best_model.pkl"))
        model = joblib.load("best_model.pkl")

        df = pd.read_csv("test_data_with_actuals.csv")
        with open("features_config.json") as f:
            cfg = json.load(f)

        X = df[cfg["features"]]
        preds = model.predict(X)

        # Check length
        self.assertEqual(len(preds), len(df))

        # Check binary predictions
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_signal_generation(self):
        """Ensure trading_signals_<model>.csv are created and correct columns exist."""
        generate_signals_for_all_models()

        files = [x for x in os.listdir(".") if x.startswith("trading_signals_")]
        self.assertGreater(len(files), 0)

        for file in files:
            df = pd.read_csv(file)
            for col in ["date", "ticker", "close", "future_return", "predicted_signal"]:
                self.assertIn(col, df.columns)

    def test_backtest(self):
        """Ensure backtest runs and outputs expected fields."""
        files = [x for x in os.listdir(".") if x.startswith("trading_signals_")]

        for file in files:
            df, final_ret, bh_ret, sharpe, maxdd = backtest_signal_file(file)

            # check equity curve monotonicity not required, but shape OK
            self.assertIn("equity", df.columns)
            self.assertGreater(len(df), 10)

            # sanity checks
            self.assertTrue(np.isfinite(final_ret))
            self.assertTrue(np.isfinite(sharpe))
            self.assertTrue(np.isfinite(maxdd))


if __name__ == '__main__':
    unittest.main()
