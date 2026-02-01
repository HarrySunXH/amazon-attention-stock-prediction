# Amazon Stock Price Prediction (Attention-based Hybrid Models)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/HarrySunXH/amazon-attention-stock-prediction?style=social)

Project showcase repo for stock price prediction using deep learning and attention-based hybrid models. Based on the paper **“Application of Attention-Based LSTM Hybrid Models for Stock Price Prediction”** (2024), and implemented primarily in `amazon_stock_prediction.ipynb`.

## Highlights
- **Data**: Amazon (AMZN) daily closing prices (last ~10 years)
- **Models**: LSTM, Self-Attention, CNN-LSTM-Attention, GRU-LSTM-Attention, CNN-BiLSTM-GRU-Attention
- **Metrics**: RMSE, R²
- **Best model (paper)**: CNN-BiLSTM-GRU-Attention (RMSE ≈ 1.0546, R² ≈ 0.9701)

**Preview**

![AMZN Closing Price](results/plots/amazon_close_price.png)

---

## Repository Structure (suggested)
```
stock-prediction-paper/
├── notebooks/
│   └── amazon_stock_prediction.ipynb          # Main notebook (primary implementation)
├── src/                                       # modularized scripts
│   ├── data_utils.py
│   ├── features.py
│   ├── models.py
│   └── train.py
├── data/
│   ├── amazon_data_close.csv
│   └── stock_data_amazon.csv
├── docs/
│   └── CODE_DOCUMENTATION.md       # Full code walkthrough
├── results/
│   ├── plots/                      # model performance charts
│   └── metrics/                    # RMSE / R2 outputs
└── README.md
```

> Tip: If you publish on GitHub, move `Amazon-Copy1.ipynb` into `notebooks/` and place datasets in `data/`.

---

## Quick Start
```bash
# (Optional) create env
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/amazon_stock_prediction.ipynb
```

### Core Dependencies
- `python>=3.9`
- `tensorflow / keras`
- `pandas`, `numpy`, `scikit-learn`
- `ta-lib` (technical indicators)
- `yfinance`, `pandas_datareader`

## Visualization

**AMZN Closing Price**

![AMZN Closing Price](results/plots/amazon_close_price.png)

**Prediction Curve (Naive Baseline)**

![Prediction Curve](results/plots/prediction_curve.png)

---

## Results & Comparison

**Model Comparison (from paper, Table 1)**

| Model | RMSE | R² | CPU Time (s) |
|---|---:|---:|---:|
| LSTM | 2.103577 | 0.881128 | 7.31 |
| Self-attention | 1.754763 | 0.917282 | 16.5 |
| CNN-LSTM-attention | 1.263602 | 0.957107 | 35.8 |
| GRU-LSTM-attention | 1.243342 | 0.958472 | 36.4 |
| CNN-BiLSTM-GRU-attention | 1.054589 | 0.970123 | 59.2 |

![RMSE & R² Bar](results/plots/model_rmse_r2_bars.png)

---

## Full Code Documentation
See: **docs/CODE_DOCUMENTATION.md**

---

## Citation
If you use this work, please cite the paper:
> Sun, X. (2024). Application of Attention-Based LSTM Hybrid Models for Stock Price Prediction. *Advances in Economics, Management and Political Sciences*, 104, 77-91.
