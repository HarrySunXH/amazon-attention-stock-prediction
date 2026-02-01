import pandas as pd


def load_close_prices(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 0]


def clean_ohlc(df: pd.DataFrame):
    cols = ['Open', 'High', 'Low', 'Close']
    for c in cols:
        if c in df.columns:
            df = df[df[c] != 0]
    return df
