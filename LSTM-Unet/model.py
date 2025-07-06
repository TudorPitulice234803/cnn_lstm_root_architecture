from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose,
                                     Dropout, concatenate, TimeDistributed, ConvLSTM2D)
from tensorflow.keras.models import Model

def lstm_unet(img_height, img_width, img_channels, time_steps):
    input_shape = (time_steps, img_height, img_width, img_channels)
    inputs = Input(shape=input_shape)  # (T, H, W, C)

    # Contracting Path with TimeDistributed
    c1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
    c1 = TimeDistributed(Dropout(0.1))(c1)
    c1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

    c2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(p1)
    c2 = TimeDistributed(Dropout(0.1))(c2)
    c2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)

    c3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(p2)
    c3 = TimeDistributed(Dropout(0.2))(c3)
    c3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

    c4 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(p3)
    c4 = TimeDistributed(Dropout(0.2))(c4)
    c4 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(c4)
    p4 = TimeDistributed(MaxPooling2D((2, 2)))(c4)

    # Bottleneck with ConvLSTM2D
    c5 = ConvLSTM2D(256, (3, 3), padding='same', return_sequences=False, activation='tanh')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Expansive Path (normal Conv2D)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4[:, -1]])  # Take last time step for skip connection
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3[:, -1]])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2[:, -1]])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1[:, -1]])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
