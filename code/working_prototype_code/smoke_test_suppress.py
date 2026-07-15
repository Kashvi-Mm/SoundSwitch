"""
SoundSwitch - Step 2 validation: STFT -> gain mask -> ISTFT round trip,
with constant (not time-varying) suppression, no classification involved.
Produces two files to listen to:
  - out_noop.wav: mask of all 1.0 -- should sound ~identical to input
  - out_traffic_const.wav: Traffic bands suppressed across the whole clip

Usage: python smoke_test_suppress.py path/to/your/audio_file.wav
"""

import sys

import soundfile as sf

from soundswitch.yamnet_utils import load_audio
from soundswitch.suppress import apply_constant_suppression, SAMPLE_RATE

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test_suppress.py path/to/audio_file.wav")
        sys.exit(1)

    waveform = load_audio(sys.argv[1])

    noop = apply_constant_suppression(waveform, muted_categories=[])
    sf.write("out_noop.wav", noop, SAMPLE_RATE)
    print("Wrote out_noop.wav (should sound ~identical to input)")

    traffic_muted = apply_constant_suppression(waveform, muted_categories=["Traffic"])
    sf.write("out_traffic_const.wav", traffic_muted, SAMPLE_RATE)
    print("Wrote out_traffic_const.wav (Traffic bands suppressed across whole clip)")
