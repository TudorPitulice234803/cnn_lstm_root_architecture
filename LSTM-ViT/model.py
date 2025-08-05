import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *

def lstm_vit():
    """
    Improved LSTM-ViT with fixes for major bottlenecks
    """
    ### HYPER PARAMS ###
    seq_len = 15
    img_height = 256
    img_width = 256
    img_channels = 3
    embedding_dim = 256  # Increased for better representation
    num_heads = 4        # More heads for better attention
    ff_dim = 512         # Increased FF dimension
    num_transformer_blocks = 4
    lstm_units = 128      # Increased LSTM capacity
    
    inputs = Input(shape=(seq_len, img_height, img_width, img_channels))
    
    def frame_encoder():
        """
        Improved frame encoder with better downsampling and skip connections
        """
        frame_input = Input(shape=(img_height, img_width, img_channels))
        
        # More gradual downsampling with batch normalization
        x = Conv2D(32, 7, strides=2, padding='same', use_bias=False)(frame_input)  # 128x128
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        skip1 = x  # Store for potential skip connection
        
        x = Conv2D(64, 3, strides=2, padding='same', use_bias=False)(x)  # 64x64
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        skip2 = x
        
        x = Conv2D(embedding_dim, 3, strides=2, padding='same', use_bias=False)(x)  # 32x32
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Additional conv layer instead of MaxPooling to preserve information
        x = Conv2D(embedding_dim, 3, strides=2, padding='same', use_bias=False)(x)  # 16x16
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Flatten spatial dims to sequence
        spatial_size = 16 * 16  # 256 patches
        x = Reshape((spatial_size, embedding_dim))(x)
        
        # Learnable positional embeddings (better than fixed)
        pos_embed_layer = Embedding(
            input_dim=spatial_size, 
            output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform'
        )
        positions = tf.range(start=0, limit=spatial_size, delta=1)
        pos_embed = pos_embed_layer(positions)
        x = x + pos_embed
        
        # Transformer blocks with improved residual connections
        for i in range(num_transformer_blocks):
            # Pre-normalization (better than post-normalization)
            x1 = LayerNormalization(epsilon=1e-6)(x)
            attn = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=embedding_dim // num_heads, 
                dropout=0.1,
                kernel_initializer='glorot_uniform'
            )(x1, x1)
            x2 = Add()([x, attn])
            
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            ff = Dense(ff_dim, activation='gelu', kernel_initializer='glorot_uniform')(x3)
            ff = Dense(embedding_dim, kernel_initializer='glorot_uniform')(ff)
            ff = Dropout(0.1)(ff)
            x = Add()([x2, ff])
        
        # Final normalization
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Reshape back to spatial format
        x = Reshape((16, 16, embedding_dim))(x)
        
        return Model(inputs=frame_input, outputs=x, name='frame_encoder')
    
    # Create the encoder
    vit_encoder = frame_encoder()
    
    # Apply encoder to each frame in the sequence
    encoded_seq = TimeDistributed(vit_encoder, name='temporal_encoding')(inputs)
    
    # MAJOR FIX: Use global average pooling instead of flattening everything
    # This reduces the sequence length dramatically
    x = TimeDistributed(GlobalAveragePooling2D(), name='spatial_pooling')(encoded_seq)
    # Now x has shape (batch_size, seq_len, embedding_dim) instead of (batch_size, seq_len, 16*16*embedding_dim)
    
    # Temporal modeling with improved LSTM
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='temporal_lstm'
    )(x)
    # x shape: (batch_size, seq_len, 2*lstm_units)
    
    # Project back to spatial features
    lstm_output_dim = 2 * lstm_units  # Bidirectional doubles the output
    x = TimeDistributed(
        Dense(16 * 16 * embedding_dim, activation='relu'),
        name='spatial_projection'
    )(x)
    x = TimeDistributed(Reshape((16, 16, embedding_dim)), name='spatial_reshape')(x)
    
    # Improved decoder with skip connections and batch norm
    x = TimeDistributed(Conv2DTranspose(
        64, 4, strides=2, padding='same', use_bias=False
    ), name='upsample_1')(x)  # 32x32
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    x = TimeDistributed(Conv2DTranspose(
        32, 4, strides=2, padding='same', use_bias=False
    ), name='upsample_2')(x)  # 64x64
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    x = TimeDistributed(Conv2DTranspose(
        16, 4, strides=2, padding='same', use_bias=False
    ), name='upsample_3')(x)  # 128x128
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    x = TimeDistributed(Conv2DTranspose(
        8, 4, strides=2, padding='same', use_bias=False
    ), name='upsample_4')(x)  # 256x256
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    
    # Final segmentation layer
    outputs = TimeDistributed(
        Conv2D(1, 1, activation='sigmoid', name='segmentation_head'),
        name='final_segmentation'
    )(x)
    
    return Model(inputs=inputs, outputs=outputs, name='LSTM_ViT')
