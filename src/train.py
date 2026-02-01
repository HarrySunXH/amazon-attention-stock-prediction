import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(series, n_steps=30):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)


def train_and_eval(model, series, n_steps=30):
    X, y = create_sequences(series, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_split=0.2, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

    preds = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2
