# Model and Strategy Comparison

This document summarizes how different machine‑learning models performed on the toy equity‐direction dataset, both in terms of **prediction accuracy** and **trading strategy performance**.

The pipeline:

- Features are engineered from daily OHLCV data (returns, moving averages, RSI, MACD, volatility, volume transforms, etc.).
- The label is **next‑day direction**: 1 if the next day’s return is positive, 0 otherwise.
- Data is split **chronologically** (no shuffling) into train and test.
- Models are trained on the processed features, then their predictions on the test set are converted into long/flat trading signals and backtested against a buy‑and‑hold benchmark.

---

## 1. Models Compared

The following models were trained on the same feature set:

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier (5‑fold cross‑validation; best fold used for trading)**

All models use only past information; features are computed from historical prices/volumes, and the label is based on a **shifted forward return**, so there is no look‑ahead bias.

---

## 2. Prediction Accuracy

On the held‑out test set (about 250 observations), the models achieve:

- **Logistic Regression:** ~0.50 accuracy  
- **Random Forest:** ~0.50 accuracy  
- **XGBoost (best fold):** ~0.54–0.55 accuracy  

Key observations:

- All models are only slightly better than random guessing (0.5), which is expected for **next‑day equity direction**, a notoriously noisy and hard‑to‑predict target.
- XGBoost consistently achieves the best accuracy among the three, benefiting from nonlinear interactions between features.
- Random Forest tends to collapse toward predicting the majority class on this dataset, giving accuracy similar to Logistic Regression but with almost no useful variation in signals.

From a pure classification standpoint, **XGBoost is the best model**, but the margin over the baselines is modest.

---

## 3. Strategy Backtesting

### 3.1. Signal Logic

For each model, predictions on the test set are converted into trading signals:

- If the predicted label is **1 (up)** → hold a **long position** (position = 1) the next day.
- If the predicted label is **0 (down)** → stay **in cash** (position = 0).
- Fixed position size and **no transaction costs** are assumed.

Strategy daily returns:

\[
r^{\text{strategy}}_t = \text{position}_{t-1} \times r^{\text{market}}_t
\]

The buy‑and‑hold benchmark simply holds a long position in the underlying asset every day.

### 3.2. Equity Curves

The plot `equity_curves.png` shows the equity curves for:

- **XGBClassifier_fold5**
- **LogisticRegression**
- **Buy & Hold**

Qualitative observations:

- **Buy & Hold** exhibits the full volatility of the underlying market—good performance in strong up trends, but also deeper drawdowns during sell‑offs.
- **Logistic Regression** produces a relatively smooth equity curve. Because its signals change infrequently and are close to always‑long or mostly‑flat, the curve steps up or stays flat for long periods.
- **XGB fold 5** shows more variation than Logistic Regression and manages to capture some medium‑term trends. In certain segments (e.g., around the middle of the test period), it tracks or slightly outperforms buy‑and‑hold while avoiding some of the deeper drawdowns.

Overall, on this toy dataset:

- **XGB fold 5** and **Logistic Regression** deliver similar final equity, with XGB showing more sensitivity to regime changes.
- **Random Forest** provides almost no useful trading signal (its equity curve is nearly flat and not plotted in the latest comparison figure), reinforcing that its predictive power here is weak.

---

## 4. Visual Diagnostics

### 4.1. Confusion Matrices / Prediction Distributions

For each model, a confusion matrix or prediction distribution can be inspected:

- Logistic Regression and XGBoost produce a mix of 0 and 1 predictions, leading to non‑trivial long/flat behavior.
- Random Forest’s predictions are heavily skewed toward the majority class (mostly predicting “down” or “no trade”), explaining the nearly flat equity curve.

These diagnostics confirm that XGBoost learns at least some pattern in the features, whereas Random Forest largely defaults to a simple majority‑class strategy.

### 4.2. Feature Importance (XGBoost)

Using XGBoost’s gain‑based feature importance, the most influential **groups of features** are:

- **Momentum features:** 1‑day and multi‑day returns, log returns.
- **Trend / mean‑reversion features:** ratio of price to short/medium moving averages, SMA spreads.
- **Volatility features:** rolling standard deviation of returns.
- **Volume dynamics:** log volume and 1‑day volume change.

These are intuitive for short‑horizon trading: the model relies mainly on recent price changes and deviations from moving‐average trends, with volatility and volume acting as secondary filters.

---

## 5. Which Model Performed Best, and Why?

### Best Model (Overall)

- **Prediction accuracy:** XGBoost (best fold) clearly leads.
- **Trading performance:** XGBoost and Logistic Regression are comparable in final equity, but XGB responds better to certain market regimes and is arguably the most interesting strategy.

**Conclusion:**  
> XGBoost (with 5‑fold cross‑validation and tuned hyperparameters) is the best model for this setup.  
> It exploits nonlinear relationships among momentum, trend, volatility, and volume features that linear models cannot fully capture, while still remaining relatively stable on this small dataset.

---

## 6. Limitations of ML in this Financial Setting

Several limitations explain why performance is modest:

1. **Label definition is extremely hard**  
   The target is next‑day direction, which is close to a random walk for many liquid assets. Even with strong models, the theoretical ceiling for accuracy is only slightly above 50%.

2. **Toy dataset & limited sample size**  
   The number of observations is small (only a few hundred test samples), which amplifies noise and makes it easy to overfit any historical quirks.

3. **Feature space is purely technical**  
   Features are based only on price and volume. No fundamentals, macro variables, order‑book information, or cross‑sectional signals are used. This strongly restricts the amount of exploitable alpha.

4. **No transaction costs or slippage**  
   Realistic trading would incur costs that could easily erase the small edge implied by ~54% accuracy or mildly positive strategy returns.

5. **Single‑asset, single‑horizon view**  
   Modern quant strategies usually diversify across many assets, horizons, and signals; here we focus on a single direction label, for a single horizon, per underlying.

---

## 7. Takeaways

Even though this is a toy project, it demonstrates a full **end‑to‑end ML trading pipeline**:

- Clean feature engineering without look‑ahead bias
- Multiple model families (linear, tree‑based, boosting)
- Cross‑validated model selection
- Conversion of predictions into trading signals
- Backtesting with equity curve comparison against buy‑and‑hold
- Qualitative interpretation of feature importance and model behavior

The key lesson is that **ML is not magic** in financial forecasting: with noisy labels and limited information, even a strong model like XGBoost can only extract a small edge. To build truly robust strategies, one needs richer data, better labels, and careful consideration of regimes, risk, and costs.
