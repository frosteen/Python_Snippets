from tensorflow.keras import layers
from tensorflow.keras.models import Model


def model_3dcnn_lstm(model_input, gestures):
    # 3DCNN Layers
    x = layers.Conv3D(32, 3, activation="relu")(model_input)
    x = layers.Conv3D(32, 3, activation="relu")(x)
    x = layers.MaxPooling3D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv3D(64, 3, activation="relu")(x)
    x = layers.Conv3D(64, 3, activation="relu")(x)
    x = layers.MaxPooling3D()(x)
    x = layers.Dropout(0.5)(x)

    # LSTM Layer
    x = layers.ConvLSTM2D(
        128,
        3,
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        recurrent_dropout=0,
        unroll=False,
        use_bias=True,
    )(x)
    x = layers.Dropout(0.5)(x)

    # Flatten to fit for Dense layer
    x = layers.Flatten()(x)

    # FC1 Layer
    x = layers.Dense(
        512,
        activation="relu",
    )(x)
    x = layers.Dropout(0.5)(x)

    # FC2 Layer
    x = layers.Dense(
        512,
        activation="relu",
    )(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(
        len(gestures),
        activation="softmax",
    )(x)

    model = Model(inputs=model_input, outputs=outputs)

    return model
