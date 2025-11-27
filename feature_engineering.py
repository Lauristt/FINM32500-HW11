import pandas as pd
import numpy as np
import json


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def run_feature_engineering():
    print("Loading data...")
    try:
        df = pd.read_csv('market_data_ml.csv')
        tickers_df = pd.read_csv('tickers-1.csv')
        with open('features_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file. {e}")
        return

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['ticker', 'date'], inplace=True)
    processed_dfs = []
    target_tickers = tickers_df['symbol'].unique()

    for ticker in target_tickers:
        sub_df = df[df['ticker'] == ticker].copy()
        if sub_df.empty:
            continue

        # Basic returns
        sub_df['return_1d'] = sub_df['close'].pct_change(1)
        sub_df['return_3d'] = sub_df['close'].pct_change(3)
        sub_df['return_5d'] = sub_df['close'].pct_change(5)

        # Log returns (nonlinear)
        sub_df['log_return_1d'] = np.log(sub_df['close']).diff()

        # Simple Moving Averages
        sub_df['sma_5'] = sub_df['close'].rolling(window=5).mean()
        sub_df['sma_10'] = sub_df['close'].rolling(window=10).mean()

        # Nonlinear: price relative to MA
        sub_df['close_over_sma5'] = sub_df['close'] / sub_df['sma_5']
        sub_df['close_over_sma10'] = sub_df['close'] / sub_df['sma_10']

        # MA spread
        sub_df['sma_spread_5_10'] = sub_df['sma_5'] - sub_df['sma_10']

        # Volatility (nonlinear)
        sub_df['volatility_5d'] = sub_df['return_1d'].rolling(5).std()

        # RSI
        sub_df['rsi_14'] = calculate_rsi(sub_df['close'], 14)

        # MACD + signal + histogram
        macd, macd_signal, macd_hist = calculate_macd(sub_df['close'])
        sub_df['macd'] = macd
        sub_df['macd_signal'] = macd_signal
        sub_df['macd_hist'] = macd_hist

        # Volume nonlinear transforms
        sub_df['log_volume'] = np.log(sub_df['volume'] + 1)
        sub_df['volume_change_1d'] = sub_df['volume'].pct_change(1)

        # y variable
        sub_df['future_return'] = sub_df['close'].pct_change(1).shift(-1)
        sub_df[config['label']] = (sub_df['future_return'] > 0).astype(int)

        # Drop NaN from rolling windows and lag
        sub_df.dropna(inplace=True)

        processed_dfs.append(sub_df)

    final_df = pd.concat(processed_dfs)

    # save
    output_file = 'processed_data.csv'
    final_df.to_csv(output_file, index=False)
    print(f"Feature engineering complete. Data saved to {output_file}")
    print(f"Features created: {config['features']}")


if __name__ == "__main__":
    run_feature_engineering()
