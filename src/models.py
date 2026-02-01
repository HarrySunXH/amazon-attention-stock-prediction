from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, TimeDistributed, Conv1D, Bidirectional, MultiHeadAttention


def build_lstm(n_steps, n_features):
    model = Sequential([
        LSTM(40, activation='sigmoid', input_shape=(n_steps, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_self_attention(n_steps, n_features):
    inputs = Input(shape=(n_steps, n_features))
    x = LSTM(40, return_sequences=True, activation='sigmoid')(inputs)
    attn = MultiHeadAttention(num_heads=2, key_dim=40)(x, x, x)
    x = LSTM(20)(attn)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_cnn_lstm_attention(n_steps, n_features):
    inputs = Input(shape=(n_steps, n_features, 1))
    x = TimeDistributed(Conv1D(32, 2, activation='relu'))(inputs)
    x = LSTM(40, return_sequences=True)(x)
    attn = MultiHeadAttention(num_heads=2, key_dim=40)(x, x, x)
    x = LSTM(40)(attn)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru_lstm_attention(n_steps, n_features):
    inputs = Input(shape=(n_steps, n_features))
    x = LSTM(32, return_sequences=True, activation='sigmoid')(inputs)
    x = LSTM(32, return_sequences=True, activation='sigmoid')(x)
    attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x, x)
    x = GRU(32, return_sequences=True, activation='sigmoid')(attn)
    x = GRU(32, activation='sigmoid')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_cnn_bilstm_gru_attention(n_steps, n_features):
    inputs = Input(shape=(n_steps, n_features, 1))
    x = TimeDistributed(Conv1D(32, 2, activation='relu'))(inputs)
    x = Bidirectional(LSTM(32, return_sequences=True, activation='sigmoid'))(x)
    attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x, x)
    x = GRU(32, activation='sigmoid')(attn)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
