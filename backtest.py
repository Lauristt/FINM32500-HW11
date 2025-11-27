import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def backtest_signal_file(signal_file):
    df = pd.read_csv(signal_file)

    df["position"] = df["predicted_signal"].shift(1).fillna(0)
    df["strategy_return"] = df["position"] * df["future_return"]
    df["equity"] = (1 + df["strategy_return"]).cumprod()
    df["buyhold"] = (1 + df["future_return"]).cumprod()

    final_return = df["equity"].iloc[-1] - 1
    final_bh = df["buyhold"].iloc[-1] - 1

    sharpe = np.sqrt(252) * df["strategy_return"].mean() / (df["strategy_return"].std() + 1e-9)
    maxdd = (df["equity"].cummax() - df["equity"]).max()

    return df, final_return, final_bh, sharpe, maxdd


def backtest_all_signals():
    signal_files = [f for f in os.listdir(".") if f.startswith("trading_signals_")]
    results = []

    plt.figure(figsize=(12, 6))

    for file in signal_files:
        df, final_ret, bh_ret, sharpe, maxdd = backtest_signal_file(file)

        model_name = file.replace("trading_signals_", "").replace(".csv", "")
        results.append([model_name, final_ret, sharpe, maxdd])

        plt.plot(df["equity"], label=model_name)

    plt.plot(df["buyhold"], label="Buy & Hold", linestyle="--", color="black")
    plt.legend()
    plt.title("Equity Curve Comparison")
    plt.ylabel("Equity")
    plt.savefig("equity_curves.png")
    plt.close()

    results_df = pd.DataFrame(results, columns=["Model", "FinalReturn", "Sharpe", "MaxDrawdown"])
    results_df.to_csv("model_backtest_summary.csv", index=False)

    print("\ Backtest Summary Saved → model_backtest_summary.csv")
    print(" Equity Comparison Saved → equity_curves.png")


if __name__ == "__main__":
    backtest_all_signals()
