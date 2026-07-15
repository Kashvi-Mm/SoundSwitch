"""
SoundSwitch - Step 1b: Frame-by-frame YAMNet classification over time
------------------------------------------------------------------------
Instead of one averaged prediction for the whole clip, this prints what
YAMNet detects roughly every second, so you can see how classification
changes across the clip (e.g. traffic vs chatter vs speech at different points).

Usage: python yamnet_timeline.py path/to/your/audio_file.wav
"""

import sys

from soundswitch.yamnet_utils import (
    load_yamnet, load_class_names, load_audio, classify, YAMNET_FRAME_HOP_SECONDS,
)


def print_timeline(scores, class_names, top_n=3, seconds_per_group=2.0):
    num_frames = scores.shape[0]
    frames_per_group = max(1, int(seconds_per_group / YAMNET_FRAME_HOP_SECONDS))

    print(f"Timeline (grouped every ~{seconds_per_group:.1f}s, showing top {top_n} classes):\n")

    for start in range(0, num_frames, frames_per_group):
        end = min(start + frames_per_group, num_frames)
        group_scores = scores[start:end].mean(axis=0)  # average within this small group only

        timestamp = start * YAMNET_FRAME_HOP_SECONDS
        top_indices = group_scores.argsort()[-top_n:][::-1]

        labels = [f"{class_names[i]} ({group_scores[i]:.2f})" for i in top_indices]
        print(f"  [{timestamp:6.1f}s]  " + "  |  ".join(labels))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yamnet_timeline.py path/to/audio_file.wav")
        sys.exit(1)

    yamnet_model = load_yamnet()
    class_names = load_class_names(yamnet_model)

    audio_path = sys.argv[1]
    print(f"Loading audio: {audio_path}")
    waveform = load_audio(audio_path)
    print(f"Audio length: {len(waveform)/16000:.2f} seconds\n")

    scores, embeddings, spectrogram = classify(yamnet_model, waveform)
    print_timeline(scores, class_names, top_n=3, seconds_per_group=2.0)
