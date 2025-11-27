import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import json


def plot_confusion(model_path,
                   data_path="test_data_with_actuals.csv",
                   config_path="features_config.json"):
    print("Confusion Matrix:", model_path)
    model = joblib.load(model_path)

    df = pd.read_csv(data_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    features = cfg["features"]
    label_col = cfg["label"]

    X = df[features]
    y_true = df[label_col]
    y_pred = model.predict(X)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(os.path.basename(model_path))
    plt.show()


def plot_prediction_distribution(signal_file):
    df = pd.read_csv(signal_file)
    df["predicted_signal"].hist(bins=3)
    plt.title(f"Prediction Distribution: {signal_file}")
    plt.show()


def plot_xgb_feature_importance(model_path):
    model = joblib.load(model_path)

    if "XGB" not in model_path:
        print("Not an XGB model.")
        return

    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    plt.barh(list(importance.keys()), list(importance.values()))
    plt.title(f"XGB Feature Importance: {model_path}")
    plt.show()
