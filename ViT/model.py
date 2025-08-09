import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Reshape, Embedding, LayerNormalization, MultiHeadAttention, Add, Dense, Dropout, Conv2DTranspose

def vit():
    embedding_dim = 128       # Embedding dimension for each patch
    num_heads = 4             # Number of attention heads
    ff_dim = 256              # Feedforward network dimension
    num_transformer_blocks = 4

    inputs = Input(shape=(256, 256, 1))

    # Use CNN layers to create feature representations
    # This preserves spatial information while creating embeddings
    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs) # 128x128
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x) # 64x64
    x = Conv2D(embedding_dim, 3, strides=2, padding='same', activation='relu')(x) # 32x32

    # Flatten spatial dims to create sequences for the transformer
    batch_size = tf.shape(x)[0]
    spatial_size = 32 * 32 # 1024 sequence length
    x = Reshape((spatial_size, embedding_dim))(x)

    # Positional embeddings
    positions = tf.range(start=0, limit=spatial_size, delta=1)
    position_embeddings = Embedding(
        input_dim=spatial_size,
        output_dim=embedding_dim
    )(positions)
    x += position_embeddings

    # Transformer blocks
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

    # Decoder to upsample back to 256x256
    x = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)

    # Final segmentation layer
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    return keras.Model(inputs=inputs, outputs=outputs)