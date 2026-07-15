"""
SoundSwitch - Real-time mic demo.
True streaming overlap-add processing, one ~32ms slice at a time -- not
accumulated chunks. Runs at your audio hardware's own native sample rate
(instead of forcing 16000Hz, our internal DSP rate) to avoid CoreAudio
silently resampling, which otherwise adds substantial latency on top of
the algorithmic floor (~128ms, the analysis window's duration).

For lowest latency, use a headset/earbuds with a SINGLE combined mic+
speaker connection (one cable/plug), not separate mic and speaker
devices -- macOS adds heavy clock-synchronization buffering (200-400ms)
when duplexing across two separate device entries, which no software
tuning here can avoid. Use --list-devices to see what's available.

NOTE: running mic input and speaker output together risks audible
feedback -- test with headphones, or keep speaker volume low at first.

Usage:
  python live_demo.py --list-devices
  python live_demo.py --device GHW-123P --mute Traffic
  python live_demo.py --mute "Mechanical Hums" Traffic
  python live_demo.py                          # no-op passthrough, default device
Press Ctrl+C to stop.
"""

import argparse
import time

import numpy as np
import sounddevice as sd

from soundswitch.categories import CATEGORIES
from soundswitch.streaming import StreamingSuppressor
from soundswitch.noise_reduction import estimate_noise_profile
from soundswitch.presence_classifier import PresenceClassifier

# Mechanical Hums deliberately excluded: unlike Traffic (a discrete event
# that comes and goes) or Chatter, hum-type noise (fans, AC, and especially
# a wired headset's own electrical/cable hum) tends to be constant and
# low-level -- not the kind of thing YAMNet's Mechanical Hums label (mostly
# engines/fans/AC recordings) reliably fires on, and gating "is a constant
# noise currently happening" on live detection doesn't make much sense
# anyway. Mechanical Hums stays on the original always-on-when-muted
# frequency-band cut, no classifier involved.
PRESENCE_CATEGORIES = ["Chatter", "Traffic"]

CALIBRATION_SECONDS = 1.5

# Audio driver callback size -- deliberately small and decoupled from the
# DSP's own hop size (which scales with sample rate, ~32ms). A callback
# size equal to the DSP hop size was measured to push CoreAudio into a
# much higher internal buffering mode (225ms vs ~110ms at smaller/auto
# sizes on this hardware) -- so we buffer internally between callbacks
# instead of forcing the audio driver to match the DSP's granularity.
CALLBACK_BLOCKSIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(description="SoundSwitch real-time mic demo")
    parser.add_argument(
        "--mute", nargs="*", default=[], metavar="CATEGORY",
        help=f"Categories to mute continuously. Choices: {', '.join(CATEGORIES)}",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device name (substring match) or index to use for both input and output. "
             "Prefer a headset with ONE combined mic+speaker connection for lowest latency.",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices and exit",
    )
    parser.add_argument(
        "--reduce-noise", action="store_true",
        help=f"Capture a ~{CALIBRATION_SECONDS}s background-noise profile at startup "
             "(stay quiet during capture) and continuously subtract it from the live audio.",
    )
    parser.add_argument(
        "--debug-chatter", action="store_true",
        help="Print live loudness/reference/ratio_db values for Chatter ducking (~every 480ms).",
    )
    parser.add_argument(
        "--force-chatter-duck", action="store_true",
        help="Debug: bypass amplitude/classifier detection and always duck when Chatter is "
             "muted, so you can A/B hear the duck amount itself regardless of whether real "
             "audio trips the detectors.",
    )
    return parser.parse_args()


def resolve_device(name_or_index):
    if name_or_index is None:
        return sd.default.device[0], sd.default.device[1]
    try:
        idx = int(name_or_index)
        return idx, idx
    except ValueError:
        pass
    for i, info in enumerate(sd.query_devices()):
        if name_or_index.lower() in info["name"].lower():
            return i, i
    raise SystemExit(f"No device matching {name_or_index!r} found. Use --list-devices to see options.")


class HopBuffer:
    """
    Decouples the audio driver's callback size from the DSP's hop size:
    accumulates incoming audio until a full hop is available, runs
    process_hop on exactly that much, and serves output from a running
    buffer -- so the driver can use whatever callback size it likes.
    """

    def __init__(self, suppressor):
        self.suppressor = suppressor
        self.pending_input = np.zeros((0,), dtype=np.float32)
        self.ready_output = np.zeros((0,), dtype=np.float32)

    def push_and_pull(self, new_samples, frames_needed):
        self.pending_input = np.concatenate([self.pending_input, new_samples])
        hop = self.suppressor.hop_length
        while len(self.pending_input) >= hop:
            chunk, self.pending_input = self.pending_input[:hop], self.pending_input[hop:]
            self.ready_output = np.concatenate([self.ready_output, self.suppressor.process_hop(chunk)])

        if len(self.ready_output) >= frames_needed:
            out, self.ready_output = self.ready_output[:frames_needed], self.ready_output[frames_needed:]
            return out
        available = len(self.ready_output)
        out = np.zeros(frames_needed, dtype=np.float32)
        out[:available] = self.ready_output
        self.ready_output = np.zeros((0,), dtype=np.float32)
        return out


if __name__ == "__main__":
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        raise SystemExit(0)

    unknown = [c for c in args.mute if c not in CATEGORIES]
    if unknown:
        print(f"Unknown categor{'y' if len(unknown) == 1 else 'ies'}: {', '.join(unknown)}")
        print(f"Choices: {', '.join(CATEGORIES)}")
        raise SystemExit(1)

    in_dev, out_dev = resolve_device(args.device)
    in_info, out_info = sd.query_devices(in_dev), sd.query_devices(out_dev)
    print(f"Input: {in_info['name']}, Output: {out_info['name']}")
    if in_dev != out_dev and in_info["default_samplerate"] != out_info["default_samplerate"]:
        print(
            f"Warning: input's native rate ({in_info['default_samplerate']:.0f}Hz) differs from "
            f"output's ({out_info['default_samplerate']:.0f}Hz) -- one side will need resampling, "
            f"adding latency. A single combined-device headset avoids this."
        )
    samplerate = int(out_info["default_samplerate"])

    print(f"Muting: {', '.join(args.mute) if args.mute else '(none)'}")

    tracked = [c for c in PRESENCE_CATEGORIES if c in args.mute]
    presence_classifier = None
    if tracked:
        presence_classifier = PresenceClassifier(sr=samplerate, categories=tracked, debug=args.debug_chatter)
        presence_classifier.start()

    suppressor = StreamingSuppressor(
        args.mute, sr=samplerate, debug_chatter=args.debug_chatter,
        presence_classifier=presence_classifier, force_chatter_duck=args.force_chatter_duck,
    )
    print(f"Running at {samplerate}Hz (device native rate), hop={suppressor.hop_length} samples")

    if args.reduce_noise:
        print(f"Capturing background noise profile -- stay quiet for {CALIBRATION_SECONDS}s...")
        calibration_audio = sd.rec(
            int(CALIBRATION_SECONDS * samplerate), samplerate=samplerate,
            channels=1, dtype="float32", device=in_dev,
        )
        sd.wait()
        suppressor.noise_profile = estimate_noise_profile(
            calibration_audio[:, 0], suppressor.n_fft, suppressor.hop_length, suppressor.window,
        )
        print("Noise profile captured.")

    buffer = HopBuffer(suppressor)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}")
        outdata[:, 0] = buffer.push_and_pull(indata[:, 0], frames)

    with sd.Stream(
        device=(in_dev, out_dev), samplerate=samplerate, channels=1, dtype="float32",
        blocksize=CALLBACK_BLOCKSIZE, latency="low", callback=callback,
    ) as stream:
        negotiated = stream.latency
        print(f"Negotiated latency (input, output): {negotiated[0]*1000:.0f}ms, {negotiated[1]*1000:.0f}ms")
        print("Listening continuously. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if presence_classifier is not None:
                presence_classifier.stop()
