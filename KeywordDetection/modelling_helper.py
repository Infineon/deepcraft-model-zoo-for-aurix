import tensorflow as tf


class KerasModel(tf.keras.models.Model):
    def __init__(self):
        super(KerasModel, self).__init__()
        inputs = tf.keras.layers.Input(shape=(50, 40))
        x = tf.keras.layers.Conv1D(
            16, 3, activation="linear", padding="same", use_bias=False
        )(inputs)
        x = tf.keras.layers.Conv1D(
            16, 3, activation="linear", padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.Conv1D(
            16, 3, activation="linear", padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)
        x = tf.keras.layers.Conv1D(
            32, 3, activation="linear", padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.Conv1D(
            32, 3, activation="linear", padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)
        x = tf.keras.layers.Conv1D(
            32, 3, activation="linear", padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.Conv1D(
            32, 3, activation="linear", padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(2, strides=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(360, activation="linear", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(36, activation="linear", use_bias=False)(x)
        x = tf.keras.layers.Activation("softmax")(x)

        self.keras_model = tf.keras.models.Model(inputs=inputs, outputs=x)

        adam = tf.keras.optimizers.Adam(learning_rate=0.004)

        self.keras_model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

    def call(self, inputs):
        return self.keras_model(inputs)


def get_model(origin="tf"):
    model = KerasModel().keras_model
    return model
