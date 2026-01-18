# Gold Price Prediction – Multi-Model Time Series Pipeline

This project implements a **complete end-to-end gold price prediction system** using
classical time-series models, machine learning, and deep learning architectures.

## Models Implemented
- Prophet (Baseline)
- SARIMA
- XGBoost Regressor
- LSTM Neural Network
- GRU Neural Network
- Transformer (Attention-based model)

The pipeline performs **EDA → feature engineering → model training → evaluation → comparison → visualization**.

---

## Dataset
- Source: Kaggle Gold Price Dataset
- File name: `gold_price_prediction.csv`
- Required columns:
  - `Date`
  - Price column (`price`, `close`, or `gold price`)
  - Optional: `open`, `high`, `low`, `vol.`, `change %`

---

## Pipeline Overview

1. Load and clean data
2. Handle missing values
3. Perform EDA and statistical analysis
4. Feature engineering (lags, rolling stats)
5. Train/Test split (80/20)
6. Train multiple forecasting models
7. Evaluate using MAE, RMSE, MAPE, Accuracy
8. Compare all models
9. Visualize best model predictions

---

## Evaluation Metrics
- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **MAPE** – Mean Absolute Percentage Error
- **Accuracy (%)** = (1 − MAPE) × 100

---

## Deep Learning Details
- Scaling: MinMaxScaler
- Window size: 60 timesteps
- Optimizer: Adam
- Loss: Mean Squared Error
- Epochs: 20
- Batch size: 32

---

## Model Selection
The best model is automatically selected based on **lowest MAPE** and visualized using:
- Actual vs Predicted time-series plot
- Prediction scatter plot


## How to Run

```bash
pip install -r requirements.txt
python gold_price_prediction.py
