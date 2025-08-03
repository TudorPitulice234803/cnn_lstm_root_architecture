import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, LayerNormalization, Dense, Dropout, Add, Embedding, MultiHeadAttention, TimeDistributed, Bidirectional, LSTM

def lstm_vit():
    ### HYPER PARAMS ###
    seq_len = 15
    img_height = 256
    img_width = 256
    img_channels = 3
    embedding_dim = 256
    num_heads = 4
    ff_dim = 512
    num_transformer_blocks = 4
    lstm_units = 64

    inputs = Input(shape=(seq_len, img_height, img_width, img_channels))

    def frame_encoder():
        frame_input = Input(shape=(img_height, img_width, img_channels))

        x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(frame_input) # 128x128
        x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x) # 64x64
        x = Conv2D(embedding_dim, 3, strides=2, padding='same', activation='relu')(x) # 32x32

        # Flatten spatial dims to sequence
        spatial_size = 32 * 32
        x = Reshape((spatial_size, embedding_dim))(x)

        # Positional embeddings
        positions = tf.range(start=0, limit=spatial_size, delta=1)
        pos_embed = Embedding(input_dim=spatial_size, output_dim=embedding_dim)(positions)
        x += pos_embed

        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x1 = LayerNormalization(epsilon=1e-6)(x)
            attn = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=0.1)(x1, x1)
            x2 = Add()([x, attn])

            x3 = LayerNormalization(epsilon=1e-6)(x2)
            ff = Dense(ff_dim, activation='gelu')(x3)
            ff = Dense(embedding_dim)(ff)
            ff = Dropout(0.1)(ff)
            x = Add()([x2, ff])

        x = LayerNormalization(epsilon=1e-6)(x)
        x = Reshape((32, 32, embedding_dim))(x)

        return Model(inputs=frame_input, outputs=x)

    vit_encoder = frame_encoder()

    # Apply the encoder on each temporal patch
    encoded_seq = TimeDistributed(vit_encoder)(inputs)

    # Flatten spatial dims per timestep for temporal modeling
    B, T, H, W, C = tf.shape(encoded_seq)[0], seq_len, 32, 32, embedding_dim
    x = Reshape((seq_len, H * W * C))(encoded_seq) # (B, 15, 32*32*256)

    # Temporal modeling across 15 frames
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2))(x)

    # Map each time step LSTM output to image sized mask
    x = TimeDistributed(Dense(H * W * 1, activation='sigmoid'))(x)
    x = TimeDistributed(Reshape((H, W, 1)))(x)

    # --- Upsample 32x32 -> 256x256 ---
    x = TimeDistributed(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'))(x)  # 64x64
    x = TimeDistributed(Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'))(x)   # 128x128
    x = TimeDistributed(Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'))(x)   # 256x256

    # --- Final segmentation layer per frame ---
    outputs = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(x)  # (B, 15, 256, 256, 1)

    return Model(inputs=inputs, outputs=outputs)
