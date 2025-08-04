# 📈 Bitcoin Price Forecasting using Time Series Models

This project explores various statistical and machine learning approaches to forecast Bitcoin closing prices. The goal is to compare traditional time series models like ARIMA and VAR with more advanced techniques like Hybrid SARIMAX + XGBoost and Ensembles.

---

## 📊 Dataset

- **Source**: [Yahoo Finance](https://finance.yahoo.com)
- **Ticker**: BTC-USD
- **Frequency**: Daily
- **Features Used**: `Open`, `High`, `Low`, `Close`, `Volume`
- **Preprocessing**: Missing values forward-filled, all series converted to daily frequency

---

## ⚙️ Workflow

### 1. Data Exploration
- Loaded historical Bitcoin data
- Focused on `Close` price for univariate models
- Visualized trends and stationarity

### 2. Train/Test Split
- 90% for training
- 10% for testing

### 3. ARIMA Models
- Tried multiple orders: `(5,2,1)`, `(0,2,2)`, `(3,1,2)`, `(4,1,0)`
- Evaluated using RMSE
- Residual analysis showed remaining non-linear patterns

### 4. SARIMA Model
- Weekly seasonality modeled using SARIMA
- Slight improvement over plain ARIMA

### 5. VAR (Vector AutoRegression)
- Used multivariate input: `Open`, `High`, `Low`, `Volume`, `Close`
- Feature scaling using `StandardScaler`
- Granger causality & impulse response for interpretability

### 6. Hybrid SARIMAX + XGBoost
- SARIMAX modeled linear & seasonal components
- Residuals used to train XGBoost for non-linear components
- Final forecast = SARIMAX forecast + XGBoost residual correction

### 7. Ensemble Model
- Combined forecasts from Hybrid, ARIMA, and VAR
- Weighted average using inverse RMSE (and later CV RMSE)

### 8. Time Series Cross-Validation
- Applied `TimeSeriesSplit` for better robustness
- Updated weights based on cross-validated RMSEs

---

## ✅ Best Performing Model

> 🏆 **Hybrid SARIMAX + XGBoost** outperformed all individual models in terms of accuracy and generalization.

**Why it works best:**
- SARIMAX handles trends and seasonality.
- XGBoost captures non-linear residual patterns left by SARIMAX.
- Cross-validation confirms its robustness across time windows.

---

## 📈 Performance Summary

| Model                     | RMSE (Test)   | Relative Score (%) |
|--------------------------|---------------|---------------------|
| 🏅 **Hybrid (SARIMAX + XGB)** | **$8,481.54**     | — (Best)            |
| VAR (Price + Volume)     | $11,415.40     | 34.8%               |
| VAR (Full Features)      | $11,581.50     | 33.8%               |
| ARIMA(4,1,0)             | $12,259.82     | 30.0%               |
| ARIMA(3,1,2)             | $12,576.72     | 28.2%               |
| ARIMA(0,2,2)             | $13,140.26     | 24.9%               |
| ARIMA(5,2,1)             | $17,504.20     | 0.0%                |

---

### 🔁 Cross-Validated RMSEs (More Reliable)

| Model                  | Cross-Validated RMSE |
|-----------------------|----------------------|
| ✅ **Hybrid Model**     | **$11,400.00**       |
| ARIMA(4,1,0)          | $12,200.00           |
| VAR (Full Features)   | $41,824.84           |

---

## 🧠 Tech Stack

- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `statsmodels` for ARIMA, SARIMAX, VAR
- `xgboost` for residual modeling
- `scikit-learn` for cross-validation
- `yfinance` for BTC-USD price data

---

## 📦 Getting Started

```bash
git clone https://github.com/yourusername/bitcoin-price-forecast.git
cd bitcoin-price-forecast
pip install -r requirements.txt
python main.py
