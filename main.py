import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow import keras


import os
from glob import glob
import re
import keras
from keras import layers

import tensorflow as tf
import keras

# khi chạy đổi lại path của file vocab theo máy m nha
with open("C:/Users/lenovo/Downloads/vocab.txt", "r", encoding="utf-8") as f:
    vocabulary = [line.rstrip("\n") for line in f]  

char_to_num = keras.layers.StringLookup(vocabulary=vocabulary, oov_token="", mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=vocabulary, oov_token="", mask_token=None, invert=True)

def CTCLoss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # thay padding -1 bằng dummy label 0
    y_true = tf.where(y_true == -1, tf.constant(0, dtype=tf.int64), y_true)

    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    # Input length : all time steps
    input_length = tf.fill([batch_size, 1], time_steps)

    # Label length: số lượng non-padding ở mỗi sequence
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

    # thêm một lớp cho blank token
    vocab_size = len(char_to_num.get_vocabulary()) + 1
    output = layers.Dense(units=vocab_size, activation="softmax", name="output")(x)

    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=CTCLoss)
    return model

model = build_model(input_dim=129, output_dim=len(char_to_num.get_vocabulary()), rnn_units=256)

model.load_weights("C:/Users/lenovo/Downloads/best_model.weights.h5")

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # greedy search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # loop ở đây để lấy text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


frame_length = 100
frame_step = 50
fft_length = 256

def predict_single_wav1(file_path):
    target_sample_rate = 16000
    audio, sr = librosa.load(file_path, sr=target_sample_rate)  # load về sample rate = 16000 và mono channel (vì ghi âm ở điện thoại thường ở dạng stereo)

    # preprocess như ở trên 
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Tạo spectrogram 
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    # Thêm batch dimension để đưa vào model
    spectrogram = tf.expand_dims(spectrogram, axis=0)

    #  prediction 
    prediction = model.predict(spectrogram)
    decoded = decode_batch_predictions(prediction)
    
    return decoded[0]


def predict_multiple_wavs(file_paths):
    results = []
    for file_path in file_paths:
        prediction = predict_single_wav1(file_path)
        results.append((file_path, prediction))
    return results

# Lấy tất cả các file .wav trong thư mục
file_paths = glob(r"C:/Users/lenovo/Desktop/wav/*.wav")  

predictions = predict_multiple_wavs(file_paths)

for file_path, transcript in predictions:
    print(f"{file_path} → {transcript}")
