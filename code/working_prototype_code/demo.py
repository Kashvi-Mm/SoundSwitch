"""
SoundSwitch - End-to-end CLI demo.
Mutes the given categories for the whole clip: Traffic/Mechanical Hums get
their frequency bands cut throughout, Chatter gets amplitude-based ducking
throughout. No classification-gating -- muting a category means it's
suppressed for as long as it's muted, matching how the real product will
work (a category stays off for as long as the user has it unchecked).

Usage:
  python demo.py input.wav output.wav --mute Traffic
  python demo.py input.wav output.wav --mute "Mechanical Hums" Traffic
  python demo.py input.wav output.wav --mute            # no-op regression check
  python demo.py --list-categories
"""

import argparse

import soundfile as sf

from soundswitch.yamnet_utils import load_audio
from soundswitch.categories import CATEGORIES
from soundswitch.suppress import apply_constant_suppression, SAMPLE_RATE


def parse_args():
    parser = argparse.ArgumentParser(description="SoundSwitch classify+suppress demo")
    parser.add_argument("input_wav", nargs="?", help="Path to input audio file")
    parser.add_argument("output_wav", nargs="?", help="Path to write the muted output")
    parser.add_argument(
        "--mute", nargs="*", default=[], metavar="CATEGORY",
        help=f"Categories to mute for the whole clip. Choices: {', '.join(CATEGORIES)}",
    )
    parser.add_argument(
        "--list-categories", action="store_true", help="Print available categories and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_categories:
        print("Available categories:", ", ".join(CATEGORIES))
        raise SystemExit(0)

    if not args.input_wav or not args.output_wav:
        print("Usage: python demo.py input.wav output.wav --mute Traffic")
        raise SystemExit(1)

    unknown = [c for c in args.mute if c not in CATEGORIES]
    if unknown:
        print(f"Unknown categor{'y' if len(unknown) == 1 else 'ies'}: {', '.join(unknown)}")
        print(f"Choices: {', '.join(CATEGORIES)}")
        raise SystemExit(1)

    print(f"Loading audio: {args.input_wav}")
    waveform = load_audio(args.input_wav)

    print(f"Muting: {', '.join(args.mute) if args.mute else '(none)'}")
    output, _ = apply_constant_suppression(waveform, args.mute)

    sf.write(args.output_wav, output, SAMPLE_RATE)
    print(f"Wrote {args.output_wav}")
