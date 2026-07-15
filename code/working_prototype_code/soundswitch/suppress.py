"""
Core DSP: build a frequency-domain gain mask from category band configs,
apply it to an STFT, and invert back to audio.
"""

import numpy as np
import librosa

from soundswitch.frequency_bands import FREQUENCY_BANDS, TAPER_WIDTH_HZ

N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 16000

# Time constant (seconds) for smoothing the mask across STFT frames, so
# Chatter's amplitude-driven gain changes ramp rather than snapping
# abruptly (an abrupt gain jump mid-clip is audible as a click/thump).
SMOOTHING_TIME_CONSTANT = 0.3

# Chatter ducking: how far below the recent loudest speech (in dB) a
# moment has to be before it's treated as quieter background chatter
# rather than the person talking to you, and how much to duck it by.
CHATTER_THRESHOLD_DB = -0.5
CHATTER_DUCK_DB = -20
# Loudness is smoothed over this long before comparing to the threshold,
# so the decision responds to sustained level changes (like a different,
# quieter voice) rather than every syllable/word within the same voice.
CHATTER_LOUDNESS_SMOOTHING_SECONDS = 0.6
# How long a "recent loud peak" reference takes to decay back down once
# nothing that loud is happening anymore -- a fast release here means one
# loud outlier (a laugh, an emphasized word) doesn't keep everything after
# it looking artificially "quiet by comparison" for very long.
CHATTER_REFERENCE_RELEASE_SECONDS = 1.0


def db_to_linear(db):
    return 10 ** (db / 20)


def band_gain_vector(freqs, low_hz, high_hz, gain_db, taper_hz=TAPER_WIDTH_HZ):
    """
    Real-valued gain per FFT frequency bin: 1.0 outside the band, gain_db
    (converted to linear) inside it, with a raised-cosine taper of width
    taper_hz at each edge so the transition is smooth rather than a hard
    cutoff (a hard edge causes ringing when inverted back to a waveform).
    """
    gain_linear = db_to_linear(gain_db)
    vec = np.ones_like(freqs)

    inside = (freqs >= low_hz) & (freqs <= high_hz)
    vec[inside] = gain_linear

    def taper(edge_center, rising_into_band):
        lo, hi = edge_center - taper_hz / 2, edge_center + taper_hz / 2
        in_taper = (freqs >= lo) & (freqs <= hi)
        if not np.any(in_taper):
            return
        t = (freqs[in_taper] - lo) / (hi - lo)  # 0..1 across the taper
        cosine = 0.5 * (1 - np.cos(np.pi * t))  # 0..1 raised-cosine ramp
        ramp = cosine if rising_into_band else (1 - cosine)
        vec[in_taper] = 1.0 + ramp * (gain_linear - 1.0)

    taper(low_hz, rising_into_band=True)
    taper(high_hz, rising_into_band=False)
    return vec


def category_gain_vector(freqs, category):
    """Combined gain vector for one category's full band list (bands multiply together)."""
    vec = np.ones_like(freqs)
    for low_hz, high_hz, gain_db in FREQUENCY_BANDS.get(category, []):
        vec *= band_gain_vector(freqs, low_hz, high_hz, gain_db)
    return vec


def apply_constant_suppression(waveform, muted_categories, sr=SAMPLE_RATE, state=None):
    """
    Suppress the given categories for the WHOLE clip: Traffic/Mechanical
    Hums get their frequency bands cut throughout, and Chatter gets
    amplitude-based ducking throughout (see chatter_duck_gain) -- no
    classification-gating, no detection windows. Muting a category means
    it's suppressed for as long as it's muted, full stop.

    state: dict carried over from a previous call, if processing a
    continuous stream in chunks (e.g. live_demo.py) -- pass the returned
    new_state into the next call so Chatter's ducking and the final mask
    smoothing don't cold-start (and therefore glitch) at every chunk
    boundary. Omit/leave as None for one-shot whole-file processing.
    Returns (output, new_state).
    """
    state = state or {}
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    num_stft_frames = stft.shape[1]

    band_categories = [c for c in muted_categories if c != "Chatter"]
    mask_vector = np.ones_like(freqs)
    for category in band_categories:
        mask_vector *= category_gain_vector(freqs, category)
    mask = np.tile(mask_vector[:, np.newaxis], (1, num_stft_frames))

    new_state = {}
    if "Chatter" in muted_categories:
        # NOTE: pitch-clarity ducking (soundswitch/pitch_clarity.py) is
        # disabled here -- librosa.pyin fails to find clean periodicity
        # almost anywhere in this specific recording (likely noise/reverb/
        # compression), not just during chatter, which would duck nearly
        # the whole clip if enabled. Needs a cleaner test recording before
        # this can be re-enabled. See project memory for details.
        amplitude_gain, chatter_state = chatter_duck_gain(
            stft, sr=sr, hop_length=HOP_LENGTH, state=state.get("chatter"),
        )
        new_state["chatter"] = chatter_state
        mask *= amplitude_gain[np.newaxis, :]
        mask = smooth_mask(mask, sr=sr, hop_length=HOP_LENGTH, initial=state.get("mask"))
        new_state["mask"] = mask[:, -1]

    modified_stft = stft * mask

    output = librosa.istft(
        modified_stft, hop_length=HOP_LENGTH, window="hann", length=len(waveform)
    )
    return output.astype(np.float32), new_state


def smooth_mask(mask, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, time_constant=SMOOTHING_TIME_CONSTANT, initial=None):
    """
    Exponential moving average across time frames (axis=1), so gain
    changes ramp smoothly instead of snapping -- same idea as attack/
    release smoothing on an audio compressor.

    initial: previous chunk's last column, if continuing a stream across
    calls (e.g. real-time chunk-by-chunk processing) -- otherwise the EMA
    cold-starts from mask[:, 0], which is correct for one-shot whole-file
    processing but would cause a discontinuity at every chunk boundary in
    a streaming setting.
    """
    frame_seconds = hop_length / sr
    alpha = 1 - np.exp(-frame_seconds / time_constant)

    smoothed = np.empty_like(mask)
    if initial is not None:
        smoothed[:, 0] = alpha * mask[:, 0] + (1 - alpha) * initial
    else:
        smoothed[:, 0] = mask[:, 0]
    for t in range(1, mask.shape[1]):
        smoothed[:, t] = alpha * mask[:, t] + (1 - alpha) * smoothed[:, t - 1]
    return smoothed


def frame_loudness(stft):
    """Per-STFT-frame overall loudness: mean magnitude across all frequency bins."""
    return np.abs(stft).mean(axis=0)


def peak_envelope(
    values, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
    release_seconds=CHATTER_REFERENCE_RELEASE_SECONDS, initial=None,
):
    """
    Causal peak-hold-with-decay envelope (the same technique a compressor's
    sidechain uses): jumps instantly to a new peak, then decays
    exponentially afterward rather than staying pinned at that peak for a
    fixed window and then dropping off a cliff. Only looks backward in
    time (no lookahead), so this could run in a live/streaming setting.

    initial: previous chunk's last envelope value, if continuing a stream
    across calls -- otherwise the envelope cold-starts from values[0].
    """
    frame_seconds = hop_length / sr
    decay = np.exp(-frame_seconds / release_seconds)

    envelope = np.empty_like(values)
    envelope[0] = max(values[0], initial * decay) if initial is not None else values[0]
    for t in range(1, len(values)):
        envelope[t] = max(values[t], envelope[t - 1] * decay)
    return envelope


def chatter_duck_gain(
    stft, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
    release_seconds=CHATTER_REFERENCE_RELEASE_SECONDS,
    loudness_smoothing_seconds=CHATTER_LOUDNESS_SMOOTHING_SECONDS,
    threshold_db=CHATTER_THRESHOLD_DB, duck_db=CHATTER_DUCK_DB,
    state=None,
):
    """
    Broadband (all-frequency) gain per STFT frame: 1.0 normally, but
    duck_db (converted to linear) for any frame that's quieter than
    threshold_db below the recent loud-peak reference -- the idea being the
    loudest recent voice is probably the person talking directly to you,
    and anything notably quieter than that recently is probably background
    chatter, not frequency content (since Chatter and direct Speech occupy
    the same frequencies -- see categories.py).

    Loudness is smoothed first (over loudness_smoothing_seconds) so the
    decision tracks sustained level -- a different, quieter voice -- rather
    than every syllable's natural loudness dip within the SAME voice
    (which would otherwise cause audible pumping in and out). The
    reference is a decaying peak envelope, not a hard windowed max, so one
    loud outlier (a laugh, an emphasized word) doesn't keep everything
    after it looking artificially "quiet by comparison" for a fixed window.

    state: dict with "loudness" and "envelope" keys carried over from the
    previous chunk, if processing a continuous stream in pieces -- pass
    the returned new_state into the next call so the ducking decision
    doesn't cold-start (and therefore glitch) at every chunk boundary.
    Returns (duck_gain, new_state).
    """
    state = state or {}
    raw_loudness = frame_loudness(stft)
    loudness = smooth_mask(
        raw_loudness[np.newaxis, :], sr=sr, hop_length=hop_length,
        time_constant=loudness_smoothing_seconds, initial=state.get("loudness"),
    )[0]

    reference = peak_envelope(
        loudness, sr=sr, hop_length=hop_length, release_seconds=release_seconds,
        initial=state.get("envelope"),
    )

    eps = 1e-8
    ratio_db = 20 * np.log10((loudness + eps) / (reference + eps))
    duck_linear = db_to_linear(duck_db)
    duck_gain = np.where(ratio_db < threshold_db, duck_linear, 1.0)

    new_state = {"loudness": loudness[-1], "envelope": reference[-1]}
    return duck_gain, new_state


