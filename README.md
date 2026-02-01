# Amazon Stock Price Prediction (Attention-based Hybrid Models)

Project showcase repo for stock price prediction using deep learning and attention-based hybrid models. Based on the paper **“Application of Attention-Based LSTM Hybrid Models for Stock Price Prediction”** (2024), and implemented primarily in `Amazon-Copy1.ipynb`.

## Highlights
- **Data**: Amazon (AMZN) daily closing prices (last ~10 years)
- **Models**: LSTM, Self-Attention, CNN-LSTM-Attention, GRU-LSTM-Attention, CNN-BiLSTM-GRU-Attention
- **Metrics**: RMSE, R²
- **Best model (paper)**: CNN-BiLSTM-GRU-Attention (RMSE ≈ 1.0546, R² ≈ 0.9701)

---

## Repository Structure (suggested)
```
stock-prediction-paper/
├── notebooks/
│   └── Amazon-Copy1.ipynb          # Main notebook (primary implementation)
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
jupyter notebook notebooks/Amazon-Copy1.ipynb
```

### Core Dependencies
- `python>=3.9`
- `tensorflow / keras`
- `pandas`, `numpy`, `scikit-learn`
- `ta-lib` (technical indicators)
- `yfinance`, `pandas_datareader`

---

## Full Code Documentation
See: **docs/CODE_DOCUMENTATION.md**

---

## Citation
If you use this work, please cite the paper:
> Xinhao Sun (2024). Application of Attention-Based LSTM Hybrid Models for Stock Price Prediction. Proceedings of the 8th International Conference on Economic Management and Green Development.
