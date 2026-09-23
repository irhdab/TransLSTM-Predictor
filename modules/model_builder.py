import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Layer,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
import numpy as np
from types import ModuleType

class PositionalEncoding(Layer):
    def __init__(self, seq_length: int | None = None, d_model: int | None = None,
                 max_len: int = 512, **kwargs: object):
        super(PositionalEncoding, self).__init__(**kwargs)
        # Backward compat: PositionalEncoding(seq_len, d_model)
        if seq_length is not None and d_model is not None:
            max_len = max(int(seq_length), 1)
            self._init_d_model = int(d_model)
        else:
            self._init_d_model = d_model
        self.max_len = int(max_len)
        self.pos_encoding: tf.Tensor | None = None

    def build(self, input_shape) -> None:
        d_model = int(input_shape[-1])
        self.pos_encoding = self.positional_encoding(self.max_len, d_model)
        super().build(input_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len})
        return cfg

    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.pos_encoding is None:
            # Fallback for graph mode before build (uses dynamic d_model)
            d_model = tf.shape(inputs)[-1]
            pe = self.positional_encoding(self.max_len, int(inputs.shape[-1] or 64))
            return inputs + pe[:, :tf.shape(inputs)[1], :]
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder_block(inputs: tf.Tensor, config: ModuleType) -> tf.Tensor:
    """
    Create a Transformer encoder block with Dropout for regularization.
    """
    d_model = int(inputs.shape[-1])
    # key_dim = d_model // heads so total attention dim == d_model
    # (was key_dim=d_model, i.e. heads*d_model params - 8x overparameterized)
    key_dim = max(1, d_model // config.TRANSFORMER_HEADS)
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(
        num_heads=config.TRANSFORMER_HEADS,
        key_dim=key_dim
    )(inputs, inputs)
    attention_output = Dropout(config.DROPOUT_RATE)(attention_output)

    # Residual connection and layer normalization
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed-Forward Network
    ffn = Dense(config.TRANSFORMER_FF_DIM, activation=config.ACTIVATION,
                kernel_regularizer=tf.keras.regularizers.l2(getattr(config, 'L2_REG', 0.0)))(attention_output)
    ffn = Dropout(config.DROPOUT_RATE)(ffn)
    ffn = Dense(d_model)(ffn)

    # Residual connection and layer normalization
    ffn_output = Add()([attention_output, ffn])
    output = LayerNormalization(epsilon=1e-6)(ffn_output)

    return output

def create_lstm_transformer_model(seq_length: int, num_features: int, config: ModuleType) -> Model:
    """
    Create CNN-LSTM-Transformer hybrid model with multi-step output.
    """
    if seq_length < 4:
        raise ValueError(f"seq_length must be >= 4, got {seq_length}")
    inputs = Input(shape=(seq_length, num_features))

    # CNN branch for local feature extraction
    cnn_layer = Conv1D(filters=config.CONV_FILTERS, kernel_size=config.CONV_KERNEL_SIZE,
                       activation=config.ACTIVATION, padding='same')(inputs)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Dropout(config.DROPOUT_RATE)(cnn_layer)

    # Transformer branch with dynamic Positional Encoding
    # (max_len slicing handles any seq_length, odd or even)
    max_len = getattr(config, 'TRANSFORMER_MAX_LEN', 512)
    transformer_branch = PositionalEncoding(max_len=max_len)(cnn_layer)
    for _ in range(config.TRANSFORMER_LAYERS):
        transformer_branch = transformer_encoder_block(transformer_branch, config)

    # Global Average Pooling instead of Flatten
    transformer_branch = GlobalAveragePooling1D()(transformer_branch)
    transformer_branch = Dense(128, activation=config.ACTIVATION)(transformer_branch)

    # Bi-LSTM branch (takes original inputs)
    lstm_branch = Bidirectional(LSTM(config.LSTM_UNITS_1, return_sequences=True))(inputs)
    lstm_branch = Bidirectional(LSTM(config.LSTM_UNITS_2, return_sequences=False))(lstm_branch)

    # Concatenate branches
    concatenated = Concatenate()([transformer_branch, lstm_branch])

    # Dense layers
    x = Dense(config.DENSE_UNITS[0], activation=config.ACTIVATION)(concatenated)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(config.DENSE_UNITS[1], activation=config.ACTIVATION)(x)
    x = Dropout(config.DROPOUT_RATE)(x)

    # Output layer: Predict FUTURE_DAYS at once
    outputs = Dense(config.FUTURE_DAYS, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
