"""Optional LSTM + attention model.

This is included to make the original RNN/LSTM project stronger and more explainable.
Install requirements-ml.txt before running this file.
"""
import tensorflow as tf
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


def build_lstm_attention_model(vocab_size: int, num_classes: int, max_len: int = 300):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, 128)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
