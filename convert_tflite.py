import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten

import os
from glob import glob
import re
import keras
from keras import layers

import tensorflow as tf
import keras


with open("vocab.txt", "r", encoding="utf-8") as f:
    vocabulary = [line.rstrip("\n") for line in f]  # Không dùng .strip()


char_to_num = keras.layers.StringLookup(vocabulary=vocabulary, oov_token="", mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=vocabulary, oov_token="", mask_token=None, invert=True)

def CTCLoss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # padding -1 cho các nhãn dummy 0 
    y_true = tf.where(y_true == -1, tf.constant(0, dtype=tf.int64), y_true)

    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    # Input lenght là tất các time stepssteps
    input_length = tf.fill([batch_size, 1], time_steps)

    # Label length đếm các giá trị không đệm trong mỗi chuỗi
    label_length = tf.math.reduce_sum(tf.cast(y_true != 0, tf.int64), axis=1, keepdims=True)

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    input_spectrogram = layers.Input((None, input_dim), name="source")
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    x = layers.Conv2D(32, [11, 31], [2, 2], padding="same", use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    x = layers.Conv2D(32, [11, 21], [1, 2], padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(rnn_units, return_sequences=True, reset_after=True, name=f"gru_{i}", implementation=2)
        x = layers.Bidirectional(recurrent, name=f"bidirectional_{i}")(x)
        if i < rnn_layers:
            x = layers.Dropout(0.5)(x)
    x = layers.Dense(rnn_units * 2, name="dense_intermediate")(x)
    x = layers.ReLU(name="dense_relu")(x)
    x = layers.Dropout(0.5)(x)

    # thêm một layer cho blank token (for CTC loss)
    vocab_size = len(char_to_num.get_vocabulary()) + 1
    output = layers.Dense(units=vocab_size, activation="softmax", name="output")(x)

    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=CTCLoss)
    return model


model = build_model(input_dim=129, output_dim=len(char_to_num.get_vocabulary()) , rnn_units=256)

model.load_weights("C:/Users/lenovo/Desktop/best_model.weights.h5")


# Lưu lại model chuẩn dạng SavedModel
model.export("C:/Users/lenovo/Desktop/saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/lenovo/Desktop/saved_model")

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_enable_resource_variables = True
converter.allow_custom_ops = True
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
with open("C:/Users/lenovo/Desktop/fixed_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Convert từ SavedModel thành công.")
