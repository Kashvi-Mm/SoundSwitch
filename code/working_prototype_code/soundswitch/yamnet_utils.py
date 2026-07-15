"""
Shared YAMNet loading/classification helpers, extracted from yamnet_test.py
and yamnet_timeline.py so new scripts don't re-implement this each time.
"""

import csv

import numpy as np
import librosa
import tensorflow_hub as hub

YAMNET_FRAME_HOP_SECONDS = 0.48  # 0.96s window, 50% hop


def load_yamnet():
    print("Loading YAMNet model...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("YAMNet loaded.\n")
    return model


def load_class_names(yamnet_model):
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            class_names.append(row[2])  # display_name column
    return class_names


def load_audio(path):
    """YAMNet requires: mono, 16kHz, float32, range [-1, 1]."""
    waveform, sr = librosa.load(path, sr=16000, mono=True)
    return waveform.astype(np.float32)


def classify(yamnet_model, waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return scores.numpy(), embeddings.numpy(), spectrogram.numpy()
