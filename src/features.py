import pandas as pd


def add_lag(data: pd.DataFrame, col: str, lags):
    for lag in lags:
        data[f"{col}_lag:{lag}"] = data[col].shift(lag)
    return data


def rolling_stats(data: pd.DataFrame, col: str, window: int):
    data[f"{col}_mean"] = data[col].rolling(window).mean()
    data[f"{col}_std"] = data[col].rolling(window).std()
    return data
