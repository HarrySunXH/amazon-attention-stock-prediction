# Code Documentation — Amazon Stock Price Prediction

This document provides a full walkthrough of the **amazon_stock_prediction.ipynb** pipeline, plus code templates for **all models mentioned in the paper**. If a model isn’t fully present in the notebook, a clean reference implementation is included here.

---

## 1. Project Objective
Predict Amazon stock closing prices using attention-based deep learning models and compare multiple architectures.

**Paper models:**
- LSTM
- Self-Attention
- CNN-LSTM-Attention
- GRU-LSTM-Attention
- CNN-BiLSTM-GRU-Attention

**Notebook models (Amazon-Copy1.ipynb) include:**
- Baseline Naive Prediction
- Vanilla LSTM
- Stacked LSTM
- LSTM-CNN (LRCN)
- Bidirectional LSTM (BiLSTM)
- LSTM-GRU
- Attention / Self-Attention variants

---

## 2. Data Pipeline

### 2.1 Data Sources
- **amazon_data_close.csv** — historical close prices (prepared file)
- **stock_data_amazon.csv** — downloaded via `yfinance` API inside the notebook

### 2.2 Data Download (Notebook Code)
```python
from pandas_datareader.data import DataReader
import yfinance as yf
from datetime import datetime, timedelta

tech_list = ['AMZN']
end = datetime.now() - timedelta(days=2)
start = datetime(end.year - 10, end.month, end.day)

stock_data = {stock: yf.download(stock, start, end) for stock in tech_list}
stock_data['AMZN'].to_csv('stock_data_amazon.csv')
```

### 2.3 Cleaning
```python
Stock = pd.read_csv("stock_data_amazon.csv", index_col=0, parse_dates=True)
Stock = Stock[(Stock.Open != 0) & (Stock.High != 0) & (Stock.Low != 0) & (Stock.Close != 0)]
```

---

## 3. Feature Engineering (from notebook)

### 3.1 Lag Features
```python
def Add_Lag(data, col, lag_list):
    for lag in lag_list:
        data[f"{col}_lag:{lag}"] = data[col].shift(lag)
    return data
```

### 3.2 Technical Indicators (TA-Lib)
```python
def Indicator(Data):
    data = Data[['Open','High','Low','Close','Volume']].copy()
    # Apply all TA-Lib indicators (excluding MAVP/OBV)
```

### 3.3 Rolling Statistics
```python
def Roll_Stats(data, col, Roll_Window):
    data[f'{col}_mean'] = data[col].rolling(Roll_Window).mean()
    data[f'{col}_std'] = data[col].rolling(Roll_Window).std()
    return data
```

### 3.4 Mutual Information Feature Selection
```python
def mutual_information_lag(Data, target_col, k_best):
    # Select top K features via mutual_info_regression
```

---

## 4. Sequence Preparation
The notebook repeatedly defines `split_sequences()` to convert time-series into supervised learning sequences.

```python
def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences)-1:
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
```

---

## 5. Models (Full Coverage)

Below are **clean, consistent reference implementations** for all models used in the paper.

### 5.1 LSTM (Baseline)
```python
model = Sequential([
    LSTM(40, activation='sigmoid', input_shape=(n_steps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

### 5.2 Self-Attention (LSTM + MultiHeadAttention)
```python
inputs = Input(shape=(n_steps, n_features))
x = LSTM(40, return_sequences=True, activation='sigmoid')(inputs)
attn = MultiHeadAttention(num_heads=2, key_dim=40)(x, x, x)
x = LSTM(20)(attn)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
```

### 5.3 CNN-LSTM-Attention
```python
inputs = Input(shape=(n_steps, n_features, 1))
x = TimeDistributed(Conv1D(32, 2, activation='relu'))(inputs)
x = LSTM(40, return_sequences=True)(x)
attn = MultiHeadAttention(num_heads=2, key_dim=40)(x, x, x)
x = LSTM(40)(attn)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
```

### 5.4 GRU-LSTM-Attention
```python
inputs = Input(shape=(n_steps, n_features))
x = LSTM(32, return_sequences=True, activation='sigmoid')(inputs)
x = LSTM(32, return_sequences=True, activation='sigmoid')(x)
attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x, x)
x = GRU(32, return_sequences=True, activation='sigmoid')(attn)
x = GRU(32, activation='sigmoid')(x)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
```

### 5.5 CNN-BiLSTM-GRU-Attention (Paper Best Model)
```python
inputs = Input(shape=(n_steps, n_features, 1))
x = TimeDistributed(Conv1D(32, 2, activation='relu'))(inputs)
x = Bidirectional(LSTM(32, return_sequences=True, activation='sigmoid'))(x)
attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x, x)
x = GRU(32, activation='sigmoid')(attn)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
```

---

## 6. Training & Evaluation

```python
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
```

Metrics reported in the paper:
- **CNN-BiLSTM-GRU-Attention**: RMSE ≈ 1.0546, R² ≈ 0.9701

---

## 7. Reproducibility Notes
- Fix random seed (numpy / tensorflow)
- Normalize with MinMaxScaler or PowerTransformer
- Align train/test split with chronological order

---

## 8. GitHub Showcase Suggestions
- Add charts in `results/plots`
- Include model comparison table (RMSE/R²)
- Provide short demo screenshot in README

---

If you want, I can also:
- Refactor notebook into clean Python scripts (`src/`)
- Add `requirements.txt`
- Produce visual model architecture diagrams
