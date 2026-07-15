"""
SoundSwitch - Web UI server.
Runs the same live streaming audio pipeline as live_demo.py, but exposes
category mute/unmute control over HTTP so ui/index.html can toggle
categories live from a browser instead of fixed --mute flags at startup.

Usage:
  python server.py --device GHW-123P
  python server.py --device GHW-123P --mute Traffic   # start with Traffic already muted
Then open http://127.0.0.1:5000 in a browser. Press Ctrl+C to stop.
"""

import argparse

import librosa
import sounddevice as sd
from flask import Flask, jsonify, request, send_from_directory

from soundswitch.categories import CATEGORIES
from soundswitch.streaming import StreamingSuppressor
from soundswitch.suppress import frame_loudness
from soundswitch.noise_reduction import estimate_noise_profile
from soundswitch.presence_classifier import PresenceClassifier
from live_demo import resolve_device, HopBuffer, CALIBRATION_SECONDS, CALLBACK_BLOCKSIZE, PRESENCE_CATEGORIES

app = Flask(__name__, static_folder=None)
suppressor = None  # set at startup, below


def parse_args():
    parser = argparse.ArgumentParser(description="SoundSwitch web UI server")
    parser.add_argument(
        "--mute", nargs="*", default=[], metavar="CATEGORY",
        help=f"Categories to start muted. Choices: {', '.join(CATEGORIES)}",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device name (substring match) or index, same as live_demo.py.",
    )
    parser.add_argument(
        "--presence-device", default=None,
        help="Separate input device (name substring or index) used ONLY to feed the "
             "Chatter/Traffic/Mechanical Hums presence classifier, distinct from --device. "
             "Use this when --device is a close-talk headset mic that doesn't pick up "
             "ambient room sound well -- e.g. point this at the MacBook's built-in mic "
             "instead. Defaults to the system default input device if omitted.",
    )
    parser.add_argument("--reduce-noise", action="store_true")
    parser.add_argument(
        "--calibrate-voice", action="store_true",
        help=f"Capture a ~{CALIBRATION_SECONDS}s recording of your normal direct-speaking "
             "voice at startup (speak normally during capture) and use it to seed the "
             "Chatter amplitude gate's 'recent loud peak' reference -- without this, that "
             "reference cold-starts from ambient audio and may never register real direct "
             "speech as louder than background chatter if no naturally loud moment happens "
             "to occur early in the session.",
    )
    parser.add_argument(
        "--input-gain", type=float, default=1.0,
        help="Uniform gain multiplier applied to incoming audio before processing (e.g. 3.0 "
             "= 3x louder). Makes quiet mic pickup audible enough to judge by ear -- does NOT "
             "change detection (amplitude-gate ratio and YAMNet's classification are both "
             "scale-invariant to a constant multiplier).",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--debug-chatter", action="store_true",
        help="Print live loudness/reference/ratio_db/classifier values for Chatter ducking.",
    )
    return parser.parse_args()


@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


@app.route("/api/state")
def get_state():
    return jsonify({"muted": sorted(suppressor.muted_categories)})


@app.route("/api/spectrogram")
def get_spectrogram():
    return jsonify({"bins": suppressor.latest_spectrum})


@app.route("/api/presence")
def get_presence():
    if suppressor.presence_classifier is None:
        return jsonify({"active": {}})
    # is_worth_showing (not is_active) -- a more lenient threshold, since
    # showing a card that then doesn't duck anything is a minor
    # inconvenience, while the suppression decision itself stays on the
    # stricter is_active threshold (soundswitch/streaming.py).
    return jsonify({"active": suppressor.presence_classifier.is_worth_showing})


@app.route("/api/toggle", methods=["POST"])
def toggle():
    data = request.get_json()
    category = data.get("category")
    muted = data.get("muted")
    if category not in CATEGORIES:
        return jsonify({"error": f"Unknown category: {category}"}), 400
    # Never mutate the set in place -- the audio thread iterates over it
    # every hop (soundswitch/streaming.py's _compute_band_mask), and an
    # in-place .add()/.discard() from this request-handling thread while
    # that iteration is in progress raises "RuntimeError: Set changed size
    # during iteration". Reassigning to a brand-new set is a single atomic
    # attribute swap instead -- the audio thread either sees the old set
    # (mid-iteration) or the new one (next hop), never a half-mutated one.
    if muted:
        suppressor.muted_categories = suppressor.muted_categories | {category}
    else:
        suppressor.muted_categories = suppressor.muted_categories - {category}
    return jsonify({"muted": sorted(suppressor.muted_categories)})


if __name__ == "__main__":
    args = parse_args()

    unknown = [c for c in args.mute if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}. Choices: {', '.join(CATEGORIES)}")

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

    # Resolve a separate input device to feed the presence classifier, if one
    # is meaningfully different from the main duplex device -- a close-talk
    # headset mic (like GHW-123P) is great for low-latency duplex but was
    # measured to pick up ambient room chatter ~10x quieter (peak amplitude)
    # than direct speech, meaning room chatter often never reaches a usable
    # signal level for classification. A room-facing mic (e.g. the laptop's
    # built-in one) doesn't have that close-talk pickup pattern.
    if args.presence_device is not None:
        presence_in_dev, _ = resolve_device(args.presence_device)
    else:
        default_in_dev = sd.default.device[0]
        presence_in_dev = default_in_dev if default_in_dev != in_dev else None
    feed_classifier_from_main_stream = presence_in_dev is None

    presence_sr = samplerate
    if not feed_classifier_from_main_stream:
        presence_in_info = sd.query_devices(presence_in_dev)
        presence_sr = int(presence_in_info["default_samplerate"])
        print(f"Presence classifier listening separately on: {presence_in_info['name']} ({presence_sr}Hz)")

    # Always start the presence classifier and always track all three
    # categories, regardless of initial --mute state -- YAMNet takes a few
    # seconds to load and needs a moment to warm up its rolling buffer, so
    # it should already be running and current by the time the user toggles
    # any category on from the browser, not start from a cold buffer then.
    presence_classifier = PresenceClassifier(
        sr=presence_sr, categories=PRESENCE_CATEGORIES, debug=args.debug_chatter,
    )
    presence_classifier.start()

    presence_stream = None
    if not feed_classifier_from_main_stream:
        def presence_callback(indata, frames, time_info, status):
            if status:
                print(f"Presence mic stream status: {status}")
            presence_classifier.push_audio(indata[:, 0])

        presence_stream = sd.InputStream(
            device=presence_in_dev, samplerate=presence_sr, channels=1, dtype="float32",
            callback=presence_callback,
        )
        presence_stream.start()

    suppressor = StreamingSuppressor(
        args.mute, sr=samplerate, presence_classifier=presence_classifier,
        debug_chatter=args.debug_chatter, feed_classifier=feed_classifier_from_main_stream,
        input_gain=args.input_gain,
    )
    print(f"Running at {samplerate}Hz (device native rate), hop={suppressor.hop_length} samples")
    print(f"Starting muted: {', '.join(args.mute) if args.mute else '(none)'}")

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

    if args.calibrate_voice:
        print(f"Capturing your direct-speaking voice -- talk normally for {CALIBRATION_SECONDS}s...")
        voice_audio = sd.rec(
            int(CALIBRATION_SECONDS * samplerate), samplerate=samplerate,
            channels=1, dtype="float32", device=in_dev,
        )
        sd.wait()
        voice_stft = librosa.stft(
            voice_audio[:, 0], n_fft=suppressor.n_fft, hop_length=suppressor.hop_length,
            window=suppressor.window,
        )
        reference_loudness = float(frame_loudness(voice_stft).max())
        suppressor.seed_chatter_reference(reference_loudness)
        print(f"Voice reference captured (loudness={reference_loudness:.5f}).")

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
        print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.")
        try:
            app.run(port=args.port, debug=False, use_reloader=False)
        finally:
            presence_classifier.stop()
            if presence_stream is not None:
                presence_stream.stop()
                presence_stream.close()
