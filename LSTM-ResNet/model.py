import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add, Conv2DTranspose,
    TimeDistributed, LSTM, Dense, ConvLSTM2D
)
from tensorflow.keras.models import Model


def lstm_resnet(img_height, img_width, img_channels, time_steps):
    """
    Model architecture for LSTM-ReNet model.
    """
    inputs = Input(shape=(time_steps, img_height, img_width, img_channels))

    # TimeDistributed ResNet Encoder
    x = TimeDistributed(Conv2D(64, (7, 7), strides=2, padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # Residual Block 1 (no downsampling)
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

    # Bottleneck - REPLACED Residual Block 3 with ConvLSTM2D
    # Spatial downsampling to match baseline ResNet's bottleneck dimensions
    x = TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # Temporal processing at bottleneck (replaces residual convolutions)
    x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same',
                   return_sequences=True, activation='tanh')(x)
    x = TimeDistributed(BatchNormalization())(x)

    # TimeDistributed Decoder
    x = TimeDistributed(Conv2DTranspose(128, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    x = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    x = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # Final prediction per frame
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(x)

    return Model(inputs=inputs, outputs=outputs)
