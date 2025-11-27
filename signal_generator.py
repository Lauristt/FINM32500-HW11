import pandas as pd
import joblib
import json
import os


def generate_signals_for_all_models(model_dir="model"):
    print(f"Scanning models in: {model_dir}")

    models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

    df = pd.read_csv("test_data_with_actuals.csv")
    with open("features_config.json", "r") as f:
        config = json.load(f)
    features = config["features"]

    for model_file in models:
        model_path = os.path.join(model_dir, model_file)
        print(f"\n Generating signals for: {model_file}")

        model = joblib.load(model_path)

        X = df[features]
        df["predicted_signal"] = model.predict(X)

        output_file = f"trading_signals_{model_file.replace('.pkl','')}.csv"
        df[["date","ticker","close","future_return","predicted_signal"]].to_csv(
            output_file, index=False
        )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    generate_signals_for_all_models()
