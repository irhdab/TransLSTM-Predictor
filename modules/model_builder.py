import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import sys
import os

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import config

def transformer_encoder_block(inputs, config):
    """
    Create a Transformer encoder block.
    
    Args:
        inputs: Input tensor
        config: Configuration object
        
    Returns:
        Output tensor
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
    Create LSTM-Transformer hybrid model for stock prediction.
    
    Args:
        seq_length (int): Length of input sequences
        num_features (int): Number of features in input
        config: Configuration object
        
    Returns:
        tf.keras.Model: Constructed model
    """
    # Input layer
    inputs = Input(shape=(seq_length, num_features))
    
    # Transformer branch
    transformer_branch = inputs
    for _ in range(config.TRANSFORMER_LAYERS):
        transformer_branch = transformer_encoder_block(transformer_branch, config)
    
    # Flatten transformer output
    transformer_branch = Flatten()(transformer_branch)
    transformer_branch = Dense(64, activation=config.ACTIVATION)(transformer_branch)
    
    # LSTM branch
    lstm_branch = LSTM(config.LSTM_UNITS_1, return_sequences=True)(inputs)
    lstm_branch = LSTM(config.LSTM_UNITS_2, return_sequences=False)(lstm_branch)
    
    # Concatenate branches
    concatenated = Concatenate()([transformer_branch, lstm_branch])
    
    # Dense layers
    x = Dense(config.DENSE_UNITS[0], activation=config.ACTIVATION)(concatenated)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(config.DENSE_UNITS[1], activation=config.ACTIVATION)(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    
    # Output layer
    outputs = Dense(1, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
