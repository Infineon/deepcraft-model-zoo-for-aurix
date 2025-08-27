# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.

# This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one 
# of its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws, 
# and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement accompanying 
# the software package from which you obtained this Software. If no license agreement applies, then any use, reproduction, modification, 
# translation, or compilation of this Software is prohibited without the express written permission of Infineon.

# Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES 
# SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make changes to the Software 
# without notice. You are responsible for properly designing, programming, and testing the functionality and safety of your intended application 
# of the Software, as well as complying with any legal requirements related to its use. Infineon does not guarantee that the Software will be 
# free from intrusion, data theft or loss, or other breaches ("Security Breaches"), and Infineon shall have no liability arising out of any 
# Security Breaches. Unless otherwise explicitly approved by Infineon, the Software may not be used in any application where a failure of the 
# Product or any consequences of the use thereof can reasonably be expected to result in personal injury.


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
