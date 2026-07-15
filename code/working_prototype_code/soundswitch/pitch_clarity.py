"""
Pitch-clarity-based chatter detection: an alternative to amplitude ducking
that doesn't need one voice to be louder than another. A single voice
speaking directly has a stable, continuously trackable pitch (fundamental
frequency). Overlapping/background chatter has multiple pitches at once,
which confuses pitch tracking into an unstable, jumpy, or low-confidence
f0 estimate -- so "how clean is the pitch tracking right now" is a proxy
for "is this one clear voice or overlapping background voices," regardless
of how loud either is.
"""

import numpy as np
import librosa

PYIN_FMIN = 65   # ~C2, covers low male voices
PYIN_FMAX = 500  # covers most adult/child speech fundamentals

# How much a frame-to-frame pitch jump (in octaves) it takes to count as
# "fully unstable" -- smaller jumps are normal pitch movement within
# fluent speech (intonation), larger jumps suggest the tracker latched
# onto a different, overlapping voice.
PITCH_JUMP_OCTAVES_FOR_FULL_INSTABILITY = 0.5

# Clarity is smoothed over this long (per the original plan's "1-2s rolling
# window") so natural intonation shifts in fluent direct speech aren't
# mistaken for the instability caused by real overlapping chatter.
CLARITY_SMOOTHING_SECONDS = 1.5
CLARITY_THRESHOLD = 0.5
CLARITY_DUCK_DB = -12


def compute_pitch_clarity(waveform, sr, frame_length, hop_length):
    """
    Per-frame clarity score in [0, 1]: voiced-ness times pitch stability.
    Low clarity = no clear single pitch to track (silence, noise, or
    multiple overlapping voices); high clarity = one voice, cleanly tracked.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        waveform, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )

    log_f0 = np.log2(f0)
    jump = np.abs(np.diff(log_f0, prepend=log_f0[0]))
    jump = np.nan_to_num(jump, nan=1.0)  # no trackable pitch here -> treat as a big jump
    stability = np.clip(1.0 - jump / PITCH_JUMP_OCTAVES_FOR_FULL_INSTABILITY, 0.0, 1.0)

    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
    return voiced_prob * stability


def clarity_duck_gain(
    waveform, sr, frame_length, hop_length,
    smoothing_seconds=CLARITY_SMOOTHING_SECONDS,
    threshold=CLARITY_THRESHOLD, duck_db=CLARITY_DUCK_DB,
):
    """
    Broadband gain per frame: 1.0 when pitch tracking is clear (one
    direct voice), duck_db when it's unstable/unclear for a sustained
    stretch (likely overlapping background chatter).
    """
    from soundswitch.suppress import smooth_mask, db_to_linear

    clarity = compute_pitch_clarity(waveform, sr, frame_length, hop_length)
    smoothed = smooth_mask(
        clarity[np.newaxis, :], sr=sr, hop_length=hop_length, time_constant=smoothing_seconds
    )[0]

    duck_linear = db_to_linear(duck_db)
    return np.where(smoothed < threshold, duck_linear, 1.0)
