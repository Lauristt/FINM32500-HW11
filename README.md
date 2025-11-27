# Assignment 11: Machine Learning in Financial Forecasting and Strategy

## Overview
This project implements a full machine-learning pipeline for financial forecasting and trading strategy evaluation, including feature generation, model training, signal generation, backtesting, and CI-driven validation.

## Pipeline Components
- Feature engineering with technical indicators and nonlinear transforms  
- ML models: Logistic Regression, Random Forest, XGBoost (5-fold CV)  
- Out-of-sample prediction and trading signal generation  
- Strategy backtesting with equity curve, Sharpe ratio, drawdown  
- Model comparison with performance visualization  
- Unit tests validating pipeline integrity  
- GitHub Actions CI for automated testing

## Project Structure
```
feature_engineering.py
train_model.py
signal_generator.py
backtest.py
workflows/
    test_pipeline.py
comparison.md
requirements.txt
.github/workflows/ci.yaml
```

## Feature Engineering
Generated predictors include:
- return_1d, return_3d, return_5d  
- log returns  
- sma_5, sma_10  
- close/SMA ratios  
- SMA spread  
- volatility_5d  
- RSI-14  
- MACD, signal, histogram  
- log(volume)  
- volume change  

Label:
```
direction = 1 if next_day_return > 0 else 0
```
All features rely strictly on past data.

## Model Training
Models trained:
- LogisticRegression  
- RandomForestClassifier  
- XGBClassifier with 5-fold cross-validation  

Outputs:
```
model/LogisticRegression.pkl
model/RandomForestClassifier.pkl
model/XGBClassifier_fold*.pkl
best_model.pkl
test_data_with_actuals.csv
```

## Signal Generation
Each model produces:
```
trading_signals_<model>.csv
```
Signals:
- 1 → long  
- 0 → flat  

## Backtesting
Backtest engine computes:
- strategy_return  
- equity curve  
- buy-and-hold baseline  
- Sharpe ratio  
- max drawdown  

Outputs:
```
equity_curves.png
model_backtest_summary.csv
```

## Model Comparison
Detailed in `comparison.md`, covering:
- model accuracy  
- financial performance  
- equity curves  
- predictions & behavior  
- ML limitations in financial forecasting  

## Unit Tests
Located in:
```
workflows/test_pipeline.py
```
Tests validate:
- feature generation  
- no look-ahead bias  
- model training  
- signal generation  
- backtesting logic  

## Running the Full Pipeline
Install:
```
pip install -r requirements.txt
```

Execute:
```
python feature_engineering.py
python train_model.py
python signal_generator.py
python backtest.py
```

Run tests:
```
pytest workflows/test_pipeline.py -q
```

## Summary
This assignment demonstrates a professional-style ML forecasting and trading framework with reproducible results, testing, and CI integration.
