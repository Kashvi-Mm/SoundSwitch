"""
SoundSwitch - one-off diagnostic: record a few seconds live and show what
YAMNet actually thinks it is (full top-label breakdown), not just the
Chatter/Hubbub score. For when the Chatter classifier reports raw_score
near 0 and it's unclear whether that's because the room audio is too
quiet/indistinct, or something else entirely.

Usage:
  python diagnose_audio.py --device GHW-123P --seconds 5
"""

import argparse

import numpy as np
import sounddevice as sd
import librosa

from live_demo import resolve_device
from soundswitch.yamnet_utils import load_yamnet, load_class_names, classify

WATCH_LABELS = ["Chatter", "Hubbub, speech noise, speech babble", "Crowd", "Speech"]

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--top", type=int, default=10)
args = parser.parse_args()

in_dev, out_dev = resolve_device(args.device)
in_info = sd.query_devices(in_dev)
sr = int(in_info["default_samplerate"])
print(f"Recording {args.seconds}s from {in_info['name']} at {sr}Hz -- make noise now...")

audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype="float32", device=in_dev)
sd.wait()
audio = audio[:, 0]
print(f"Captured. Peak amplitude: {np.abs(audio).max():.4f}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")

audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio

yamnet_model = load_yamnet()
class_names = load_class_names(yamnet_model)
scores, _, _ = classify(yamnet_model, audio_16k)

mean_scores = scores.mean(axis=0)
top_idx = np.argsort(mean_scores)[::-1][:args.top]

print(f"\nTop {args.top} labels (mean score across {scores.shape[0]} frames):")
for i in top_idx:
    print(f"  {class_names[i]:<40s} {mean_scores[i]:.4f}")

print("\nChatter-relevant labels (regardless of rank):")
for label in WATCH_LABELS:
    i = class_names.index(label)
    print(f"  {label:<40s} {mean_scores[i]:.4f}")
