import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *

def lstm_vit():
    """
    LSTM-ViT with backbone matched to simple ViT for fair comparison
    """
    ### HYPER PARAMS ###
    seq_len = 15
    img_height = 256
    img_width = 256
    img_channels = 3
    embedding_dim = 256       # Same as simple ViT
    num_heads = 4             # Same as simple ViT
    ff_dim = 512              # Same as simple ViT
    num_transformer_blocks = 4 # Same as simple ViT
    lstm_units = 64           # LSTM specific
    
    inputs = Input(shape=(seq_len, img_height, img_width, img_channels))
    
    def frame_encoder():
        """
        Frame encoder that exactly matches the simple ViT backbone
        """
        frame_input = Input(shape=(img_height, img_width, img_channels))
        
        # EXACT same CNN layers as simple ViT
        x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(frame_input) # 128x128
        x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x) # 64x64
        x = Conv2D(embedding_dim, 3, strides=2, padding='same', activation='relu')(x) # 32x32
        
        # Flatten spatial dims to create sequences for the transformer
        spatial_size = 32 * 32  # 1024 sequence length
        x = Reshape((spatial_size, embedding_dim))(x)
        
        # Positional embeddings - EXACT same as simple ViT
        positions = tf.range(start=0, limit=spatial_size, delta=1)
        position_embeddings = Embedding(
            input_dim=spatial_size,
            output_dim=embedding_dim
        )(positions)
        x += position_embeddings
        
        # Transformer blocks - EXACT same as simple ViT
        for _ in range(num_transformer_blocks):
            # Multi head Self Attention
            x1 = LayerNormalization(epsilon=1e-6)(x)
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            x2 = Add()([x, attention_output])
            
            # Feed forward Network
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            ff_output = Dense(ff_dim, activation='gelu')(x3)
            ff_output = Dense(embedding_dim)(ff_output)
            ff_output = Dropout(0.1)(ff_output)
            x = Add()([x2, ff_output])
        
        # Final Normalization
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Reshape back to spatial dims (32x32)
        x = Reshape((32, 32, embedding_dim))(x)
        
        return Model(inputs=frame_input, outputs=x, name='vit_encoder')
    
    # Create the encoder (same as simple ViT)
    vit_encoder = frame_encoder()
    
    # Apply encoder to each frame in the sequence
    encoded_seq = TimeDistributed(vit_encoder, name='temporal_encoding')(inputs)
    # Shape: (batch_size, seq_len, 32, 32, embedding_dim)
    
    # Flatten spatial dimensions for LSTM processing
    B = tf.shape(encoded_seq)[0]
    T = seq_len
    H, W, C = 32, 32, embedding_dim
    x = Reshape((T, H * W * C))(encoded_seq)  # (batch_size, seq_len, 32*32*256)
    
    # Temporal modeling with Bidirectional LSTM
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.2),
        name='temporal_lstm'
    )(x)
    # Shape: (batch_size, seq_len, 2*lstm_units)
    
    # Project LSTM output back to spatial feature size
    x = TimeDistributed(
        Dense(H * W * C, activation='relu'),
        name='spatial_projection'
    )(x)
    
    # Reshape back to spatial dimensions
    x = Reshape((T, H, W, C))(x)
    # Shape: (batch_size, seq_len, 32, 32, embedding_dim)
    
    # Decoder - EXACT same as simple ViT
    x = TimeDistributed(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'))(x)  # 64x64
    x = TimeDistributed(Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'))(x)   # 128x128
    x = TimeDistributed(Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'))(x)   # 256x256
    
    # Final segmentation layer - same as simple ViT
    outputs = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(x)
    
    return Model(inputs=inputs, outputs=outputs, name='LSTM_ViT')