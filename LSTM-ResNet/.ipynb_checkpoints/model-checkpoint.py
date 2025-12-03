import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add, Conv2DTranspose,
    TimeDistributed, LSTM, Dense, ConvLSTM2D
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
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same',
                  return_sequences=True, activation='tanh')(x)
    x = BatchNormalization()(x)

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
