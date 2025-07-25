import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add, Conv2DTranspose,
    TimeDistributed, LSTM, Flatten, Dense, Reshape
)
from tensorflow.keras.models import Model

def lstm_resnet(img_height, img_width, img_channels, time_steps):
    inputs = Input(shape=(time_steps, img_height, img_width, img_channels))

    ### TimeDistributed ResNet Encoder ###
    x = TimeDistributed(Conv2D(64, (7, 7), strides=2, padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # Residual Block 1
    shortcut = x
    x = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([shortcut, x])
    x = TimeDistributed(ReLU())(x)

    # Residual Block 2 (with downsampling)
    shortcut = TimeDistributed(Conv2D(128, (1, 1), strides=2, padding='same'))(x)
    shortcut = TimeDistributed(BatchNormalization())(shortcut)

    x = TimeDistributed(Conv2D(128, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(Conv2D(128, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = Add()([shortcut, x])
    x = TimeDistributed(ReLU())(x)

    # Residual Block 3 (with downsampling)
    shortcut = TimeDistributed(Conv2D(256, (1, 1), strides=2, padding='same'))(x)
    shortcut = TimeDistributed(BatchNormalization())(shortcut)

    x = TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(Conv2D(256, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = Add()([shortcut, x])
    x = TimeDistributed(ReLU())(x)

    ### Flatten spatial for LSTM ###
    x = TimeDistributed(Flatten())(x)  # shape: (batch, time, features)

    ### LSTM Across Time ###
    x = LSTM(256, return_sequences=True)(x)

    ### Project back to spatial format ###
    spatial_dim = (img_height // 8, img_width // 8, 256)
    x = TimeDistributed(Dense(tf.math.reduce_prod(spatial_dim)))(x)
    x = TimeDistributed(Reshape(spatial_dim))(x)

    ### TimeDistributed Decoder (Upsampling without residuals) ###
    x = TimeDistributed(Conv2DTranspose(128, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    x = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    x = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    ### Final prediction per frame ###
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(x)

    return Model(inputs=inputs, outputs=outputs)
