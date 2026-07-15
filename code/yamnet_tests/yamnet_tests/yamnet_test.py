"""
SoundSwitch - Step 1: Load YAMNet and classify an audio file
--------------------------------------------------------------
Run this first to confirm YAMNet works before building anything else.
Usage: python yamnet_test.py path/to/your/audio_file.wav
"""

import sys

from soundswitch.yamnet_utils import load_yamnet, load_class_names, load_audio, classify


def print_top_classes(scores, class_names, top_n=5):
    mean_scores = scores.mean(axis=0)
    top_indices = mean_scores.argsort()[-top_n:][::-1]
    print(f"Top {top_n} predicted classes (averaged over whole clip):")
    for i in top_indices:
        print(f"  {class_names[i]:<30} {mean_scores[i]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yamnet_test.py path/to/audio_file.wav")
        sys.exit(1)

    yamnet_model = load_yamnet()
    class_names = load_class_names(yamnet_model)
    print(f"Loaded {len(class_names)} class names.\n")

    audio_path = sys.argv[1]
    print(f"Loading audio: {audio_path}")
    waveform = load_audio(audio_path)
    print(f"Audio length: {len(waveform)/16000:.2f} seconds\n")

    scores, embeddings, spectrogram = classify(yamnet_model, waveform)
    print(f"Scores shape: {scores.shape}  (frames x 521 classes)")
    print(f"Spectrogram shape: {spectrogram.shape}\n")

    print_top_classes(scores, class_names, top_n=5)
