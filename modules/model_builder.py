import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import sys
import os

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import config

class PositionalEncoding(Layer):
    def __init__(self, seq_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(seq_length, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder_block(inputs, config):
    """
    Create a Transformer encoder block.
    """
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(
        num_heads=config.TRANSFORMER_HEADS,
        key_dim=inputs.shape[-1]
    )(inputs, inputs)
    
    # Residual connection and layer normalization
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    
    # Feed-Forward Network
    ffn = Dense(config.TRANSFORMER_FF_DIM, activation=config.ACTIVATION)(attention_output)
    ffn = Dense(inputs.shape[-1])(ffn)
    
    # Residual connection and layer normalization
    ffn_output = Add()([attention_output, ffn])
    output = LayerNormalization()(ffn_output)
    
    return output

def create_lstm_transformer_model(seq_length, num_features, config):
    """
    Create CNN-LSTM-Transformer hybrid model with multi-step output.
    """
    inputs = Input(shape=(seq_length, num_features))
    
    # CNN branch for local feature extraction
    cnn_layer = Conv1D(filters=config.CONV_FILTERS, kernel_size=config.CONV_KERNEL_SIZE, 
                       activation=config.ACTIVATION, padding='same')(inputs)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Dropout(config.DROPOUT_RATE)(cnn_layer)
    
    # Transformer branch with Positional Encoding
    transformer_branch = PositionalEncoding(seq_length // 2, config.CONV_FILTERS)(cnn_layer)
    for _ in range(config.TRANSFORMER_LAYERS):
        transformer_branch = transformer_encoder_block(transformer_branch, config)
    
    # Global Average Pooling instead of Flatten
    transformer_branch = GlobalAveragePooling1D()(transformer_branch)
    transformer_branch = Dense(128, activation=config.ACTIVATION)(transformer_branch)
    
    # LSTM branch (takes original inputs or CNN output)
    lstm_branch = LSTM(config.LSTM_UNITS_1, return_sequences=True)(inputs)
    lstm_branch = LSTM(config.LSTM_UNITS_2, return_sequences=False)(lstm_branch)
    
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
