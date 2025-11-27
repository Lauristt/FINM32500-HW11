import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models_with_5folds():

    print("Loading processed data and configs...")
    try:
        df = pd.read_csv("processed_data.csv")
        with open("features_config.json", "r") as f:
            feat_cfg = json.load(f)
        with open("model_params.json", "r") as f:
            model_cfg = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: {e}  → Run feature_engineering.py first.")
        return

    features = feat_cfg["features"]
    label = feat_cfg["label"]

    X = df[features]
    y = df[label]

    # ---- Chronological split (no shuffle) ----
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size = {len(X_train)}, Test size = {len(X_test)}")

    models = {}  # store trained models
    #lr
    if "LogisticRegression" in model_cfg:
        print("\nTraining Logistic Regression...")
        lr_params = model_cfg["LogisticRegression"]
        lr = LogisticRegression(**lr_params)
        lr.fit(X_train, y_train)
        models["LogisticRegression"] = lr
        joblib.dump(lr, "model/LogisticRegression.pkl")

    #rf
    if "RandomForestClassifier" in model_cfg:
        print("\nTraining Random Forest...")
        rf_params = model_cfg["RandomForestClassifier"]
        rf = RandomForestClassifier(**rf_params, random_state=42)
        rf.fit(X_train, y_train)
        models["RandomForestClassifier"] = rf
        joblib.dump(rf, "model/RandomForestClassifier.pkl")

    #xgb
    if "XGBClassifier" in model_cfg:
        print("\nTraining XGBoost with 5-fold CV...")

        xgb_params = model_cfg["XGBClassifier"]
        kf = KFold(n_splits=5, shuffle=False)

        fold_accuracies = []
        fold_models = []

        for fold_idx, (tr, va) in enumerate(kf.split(X_train)):
            print(f"  Fold {fold_idx+1}/5")

            X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]

            model = XGBClassifier(
                **xgb_params,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            )
            model.fit(X_tr, y_tr)

            preds_va = model.predict(X_va)
            acc = accuracy_score(y_va, preds_va)
            fold_accuracies.append(acc)
            fold_models.append(model)

            print(f"    Fold {fold_idx+1} accuracy: {acc:.4f}")

            # save each fold for strategy comparison
            joblib.dump(model, f"model/XGBClassifier_fold{fold_idx+1}.pkl")

        print(f"XGBoost 5-fold mean accuracy: {np.mean(fold_accuracies):.4f}")

        # use the best fold model for prediction
        best_fold = np.argmax(fold_accuracies)
        best_xgb = fold_models[best_fold]
        models["XGBClassifier"] = best_xgb

    print("\nModel evaluation on test set...")
    best_model = None
    best_acc = -1
    summary = []

    for name, model in models.items():
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        summary.append((name, acc))
        print(f"{name}: Test Accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(f"\nBest model: {type(best_model).__name__} (Acc = {best_acc:.4f})")
    joblib.dump(best_model, "best_model.pkl")
    test_df = df.iloc[X_test.index].copy()
    test_df.to_csv("test_data_with_actuals.csv", index=False)
    print("Saved: test_data_with_actuals.csv")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_models_with_5folds()
